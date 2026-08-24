import argparse
import csv
import os
import time
import glob
from datetime import timedelta
from typing import Dict, List
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.dataset.vqa_dataset import VQADataset



class ModelInference:
    """Distributed model inference class for evaluation"""
    
    def __init__(
        self,
        data_loader,
        base_model_path: str,
        processor_name: str,
        max_new_tokens: int,
        split: str,
        peft_model_path: str = None,
        num_return_sequences: int = 1,
        on_reasoning: bool = False,
        accelerator: Accelerator = None,
        with_augmentation: bool = False,
    ):
        self.max_new_tokens = max_new_tokens
        self.split = split
        self.num_return_sequences = num_return_sequences
        self.on_reasoning = on_reasoning
        self.accelerator = accelerator
        self.with_augmentation = with_augmentation
        
        # Set device based on accelerator process index
        self.device = f"cuda:{self.accelerator.process_index}"
        
        # Store model paths and parameters - models will be loaded on demand
        self.base_model_path = base_model_path
        self.processor_name = processor_name
        self.peft_model_path = peft_model_path
        
        # Qwen2.5VL model will be loaded during inference, not during initialization
        self.model = None
        self.processor = None
        
        self.data_loader = data_loader
    
    def _load_vlm(self):
        """Load a vision-language model for inference (Qwen2.5-VL, Qwen3-VL, etc.)."""
        if self.model is not None:
            return  # Already loaded
        
        rank_info = f"[Rank {self.accelerator.process_index}] "
        print(f"{rank_info}Loading VLM from: {self.base_model_path}")
        if self.peft_model_path:
            print(f"{rank_info}With PEFT adapter: {self.peft_model_path}")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_path,
            device_map=None,
            dtype=torch.bfloat16,
            # local_files_only=True,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        
        if self.peft_model_path:
            self.model = PeftModel.from_pretrained(self.model, self.peft_model_path, strict=False)
        
        self.processor = AutoProcessor.from_pretrained(
            self.processor_name,
            max_pixels=512 * 28 * 28,
            # local_files_only=True,
        )
        self.processor.tokenizer.padding_side = 'left'
    
    def _preprocess_messages(self, batch: List[Dict]) -> List[List[Dict]]:
        """Construct messages directly in Qwen format"""
        return [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image_path"]},
                        {"type": "text", "text": item["question"]},
                    ],
                }
            ]
            for item in batch
        ]
    
    def _generate_responses(self, batch: List[Dict]) -> List[str]:
        messages = self._preprocess_messages(batch)
        
        # Prepare for batch inference - pass all messages at once with padding=True
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            if self.num_return_sequences == 1:
                # Use greedy decoding for single sequence generation
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,  # Greedy decoding
                    num_return_sequences=self.num_return_sequences,
                    max_new_tokens=self.max_new_tokens,
                )
            else:
                # Use sampling for multiple sequence generation
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    # top_k=50,
                    num_return_sequences=self.num_return_sequences,
                    max_new_tokens=self.max_new_tokens,
                )

        # Trim input_ids from generated_ids
        # When num_return_sequences > 1, generated_ids shape is [batch_size * num_return_sequences, seq_len]
        # inputs["input_ids"] needs to be repeated accordingly
        if self.num_return_sequences > 1:
            # Repeat each input_ids num_return_sequences times
            input_ids_expanded = inputs["input_ids"].repeat_interleave(self.num_return_sequences, dim=0)
        else:
            input_ids_expanded = inputs["input_ids"]
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_expanded, generated_ids)
        ]

        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputs
    
    
    def run_inference(self, output_path: str, batch_size: int = 16, save_every_n_batch : int = 4) -> str:
        """Run distributed inference on the dataset and save results"""
        start_time = time.time()
        
        rank_info = f"[Rank {self.accelerator.process_index}] "
        print(f"{rank_info}Starting inference on {self.split} set...")
        
        # Store original output path before adding rank suffix
        original_output_path = output_path
        
        try:
            # Load VLM for inference
            self._load_vlm()
            
            # Create rank-specific output file
            base_name, ext = os.path.splitext(output_path)
            rank_output_path = f"{base_name}_rank_{self.accelerator.process_index}{ext}"
            print(f"{rank_info}Using rank-specific output file: {rank_output_path}")
            
            # Check if output file already exists and throw error
            if os.path.exists(rank_output_path):
                raise FileExistsError(f"{rank_info}Output file already exists: {rank_output_path}. Please remove it or use a different output path.")
            
            # Initialize file with header
            self._initialize_csv_file(rank_output_path)
            
            batch_results = []  # Store current batch results
            total_sequences = 0  # Track total sequences processed
            batch_count = 0
            
            # Process batches from data loader
            for batch in self.data_loader:
                # Run inference on batch
                outputs = self._generate_responses(batch)
                
                # Organize results
                # When num_return_sequences > 1, outputs shape is [batch_size * num_return_sequences]
                # When augmentation is enabled, num_return_sequences must be 1
                if self.num_return_sequences > 1:
                    # Expand batch items to match outputs (batch_size * num_return_sequences outputs)
                    expanded_items = []
                    for item in batch:
                        for seq_idx in range(self.num_return_sequences):
                            expanded_items.append(item)
                    
                    for idx, (output, item) in enumerate(zip(outputs, expanded_items)):
                        augmentation_type = item.get("augmentation_type", "original")
                        # Calculate sequence_id from position in expanded list
                        sequence_id = idx % self.num_return_sequences
                        
                        result = {
                            "id": item["id"],
                            "qid": item["qid"],
                            "sequence_id": sequence_id,
                            "augmentation_type": augmentation_type,
                            "question": item["question"],
                            "answer": item["answer"],
                            "output": output,
                            "image_path": item["image_path"]
                        }
                        if self.on_reasoning:
                            result['reasoning'] = item['reasoning']
                        batch_results.append(result)
                else:
                    # Single sequence per item (normal case and augmentation case)
                    for output, item in zip(outputs, batch):
                        augmentation_type = item.get("augmentation_type", "original")
                        sequence_id = item.get("sequence_id", 0)
                        
                        result = {
                            "id": item["id"],
                            "qid": item["qid"],
                            "sequence_id": sequence_id,
                            "augmentation_type": augmentation_type,
                            "question": item["question"],
                            "answer": item["answer"],
                            "output": output,
                            "image_path": item["image_path"]
                        }
                        if self.on_reasoning:
                            result['reasoning'] = item['reasoning']
                        batch_results.append(result)
                
                total_sequences += len(outputs)
                batch_count += 1
                
                # Progress update and save
                if batch_count % save_every_n_batch == 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    total_samples = total_sequences // max(1, self.num_return_sequences)
                    print(f"{rank_info}Processed {total_sequences} sequences ({total_samples} samples). Time: {elapsed_minutes:.1f} min")
                    self._append_results(batch_results, rank_output_path)
                    batch_results = []  # Clear after saving
            
            # Save any remaining results
            if batch_results:
                self._append_results(batch_results, rank_output_path)
            
            elapsed_minutes = (time.time() - start_time) / 60
            total_samples = total_sequences // max(1, self.num_return_sequences)
            print(f"{rank_info}Inference completed! {total_samples} samples ({total_sequences} sequences) in {elapsed_minutes:.1f} min")
            print(f"{rank_info}Results saved to: {rank_output_path}")
        
        finally:
            pass
        
        # Wait for all processes to finish before merging
        self.accelerator.wait_for_everyone()
        
        # Merge all rank-specific files into a single output file
        final_output_path = self._merge_rank_files(original_output_path)
        
        return final_output_path
    
    def _initialize_csv_file(self, output_path: str):
        """Initialize CSV file with header"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create empty file with header
        with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
            # We'll write the header when we have the first batch
            pass
    
    def _append_results(self, results: List[Dict], output_path: str):
        """Append results to CSV file"""
        if not results:
            return
            
        # Check if file is empty (no header written yet)
        file_exists = os.path.exists(output_path)
        file_is_empty = file_exists and os.path.getsize(output_path) == 0
        
        with open(output_path, "a", newline='', encoding="utf-8") as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is empty
            if file_is_empty or not file_exists:
                writer.writeheader()
            
            writer.writerows(results)
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Save results to CSV file (legacy method for compatibility)"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
    
    def _merge_rank_files(self, original_output_path: str) -> str:
        """Merge all rank-specific output files into a single file"""
        if self.accelerator.process_index != 0:
            return original_output_path  # Only rank 0 does the merging
        
        rank_info = f"[Rank {self.accelerator.process_index}] "
        base_name, ext = os.path.splitext(original_output_path)
        
        # Find all rank-specific files
        rank_file_pattern = f"{base_name}_rank_*{ext}"
        rank_files = sorted(glob.glob(rank_file_pattern))
        
        if not rank_files:
            print(f"{rank_info}Warning: No rank-specific files found to merge")
            return original_output_path
        
        print(f"{rank_info}Merging {len(rank_files)} rank-specific files into: {original_output_path}")
        
        # Check if final output file already exists
        if os.path.exists(original_output_path):
            raise FileExistsError(
                f"{rank_info}Final output file already exists: {original_output_path}. "
                "Please remove it or use a different output path."
            )
        
        # Merge all rank files
        fieldnames = None
        total_rows = 0
        
        with open(original_output_path, "w", newline='', encoding="utf-8") as outfile:
            writer = None
            
            for rank_file in rank_files:
                if not os.path.exists(rank_file) or os.path.getsize(rank_file) == 0:
                    print(f"{rank_info}Warning: Skipping empty or missing file: {rank_file}")
                    continue
                
                with open(rank_file, "r", newline='', encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)
                    
                    # Skip if file has no header/fieldnames
                    if reader.fieldnames is None:
                        print(f"{rank_info}Warning: No header found in {rank_file}, skipping")
                        continue
                    
                    # Set fieldnames from first file
                    if fieldnames is None:
                        fieldnames = reader.fieldnames
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
                    elif reader.fieldnames != fieldnames:
                        print(f"{rank_info}Warning: Fieldnames mismatch in {rank_file}, skipping")
                        continue
                    
                    # Write all rows from this rank file
                    rows_written = 0
                    for row in reader:
                        writer.writerow(row)
                        rows_written += 1
                    
                    total_rows += rows_written
                    print(f"{rank_info}Merged {rows_written} rows from {os.path.basename(rank_file)}")
        
        print(f"{rank_info}Successfully merged {total_rows} total rows into {original_output_path}")
        
        # Delete rank-specific files after successful merge
        for rank_file in rank_files:
            try:
                os.remove(rank_file)
                print(f"{rank_info}Deleted rank-specific file: {os.path.basename(rank_file)}")
            except Exception as e:
                print(f"{rank_info}Warning: Could not delete {rank_file}: {e}")
        
        return original_output_path


def main():
    parser = argparse.ArgumentParser(description="Model Inference for Evaluation")
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="Path to the base model directory")
    parser.add_argument("--processor_name", type=str, default=None,
                       help="Processor name/path. Defaults to base_model_path when omitted.")
    parser.add_argument("--peft_model_path", type=str, default=None,
                       help="Path to PEFT adapter model (optional)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/emzed/projects/aip-dolatab6/emzed/med-align/experiments/",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--split", type=str, choices=["test", "train", "validation"], required=True,
                       help="Split to use for inference")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help="Number of sequences to return for each sample")
    parser.add_argument("--on_reasoning", action="store_true",
                       help="Enable reasoning mode for the dataset")
    parser.add_argument("--with_augmentation", action="store_true",
                       help="Enable augmentation (original, blur_medsam, noise_medsam, mask_medsam)")
    parser.add_argument("--augmented_images_dir", type=str, default=None,
                       help="Directory containing pre-generated augmented images (required if --with_augmentation is set)")
    parser.add_argument("--distributed_timeout_minutes", type=int, default=60,
                       help="Timeout in minutes for distributed barrier/store (default 60). Increase if rank 0 is slow.")

    args = parser.parse_args()
    
    # Initialize accelerator for distributed inference (always enabled).
    # Use a long timeout so rank 1 doesn't give up waiting for rank 0 (PyTorch default is 10 min).
    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(minutes=args.distributed_timeout_minutes))
        ]
    )
    print(f"Initialized distributed inference with {accelerator.num_processes} processes")
    
    peft_name = "None"
    if args.peft_model_path:
        peft_path = Path(args.peft_model_path)
        # allowed_peft_names = ["dpo", "rrpo", "sft"]
        # for name in allowed_peft_names:
        #     if name in peft_path.parent.name.lower():
        #         peft_name = name
        #         break
        # else:
        #     raise ValueError(
        #         f"Parent folder name '{peft_path.parent.name}' must contain one of: {', '.join(allowed_peft_names)}"
        #     )
    
        peft_name = peft_path.parent.name.lower()
    
    
    # if args.num_return_sequences != 1 and args.split == "test":
    #     raise ValueError("num_return_sequences != 1 is only supported in non-test splits")
    
    if args.with_augmentation:
        if not args.augmented_images_dir:
            raise ValueError("--augmented_images_dir must be provided when --with_augmentation is set")
    
    if args.with_augmentation and args.num_return_sequences != 1:
        raise ValueError("--num_return_sequences must be 1 when --with_augmentation is set")
    
    aug_suffix = "_aug" if args.with_augmentation else ""
    output_filename = f"{peft_name}_{args.num_return_sequences}_{args.split}{aug_suffix}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    
    dataset = VQADataset(args.dataset_name, args.split, reasoning=args.on_reasoning)
    
    # Split data across ranks for distributed inference
    total_samples = len(dataset)
    samples_per_rank = total_samples // accelerator.num_processes
    start_idx = accelerator.process_index * samples_per_rank
    end_idx = start_idx + samples_per_rank
    
    # Handle the last rank to include any remaining samples
    if accelerator.process_index == accelerator.num_processes - 1:
        end_idx = total_samples
    
    print(f"[Rank {accelerator.process_index}] Processing samples {start_idx} to {end_idx-1} ({end_idx-start_idx} samples)")
    
    # Create a custom data loader for this rank's subset
    def distributed_dataloader():
        if args.with_augmentation:
            # Load augmented images mapping
            rank_info = f"[Rank {accelerator.process_index}] "
            print(f"{rank_info}Loading augmented images from: {args.augmented_images_dir}")
            
            if not os.path.exists(args.augmented_images_dir):
                raise FileNotFoundError(f"{rank_info}Augmented images directory not found: {args.augmented_images_dir}")
            
            # Create mapping from sample_id to augmented image paths
            sample_id_to_augmented_images = {}
            augmentation_types = ["original", "blur", "noise", "mask"]
            
            for filename in os.listdir(args.augmented_images_dir):
                if not filename.endswith('.png'):
                    continue
                
                # Parse filename: {sample_id}_{augmentation_type}.png
                if '_original.png' in filename:
                    sample_id_str = filename.replace('_original.png', '')
                    augmentation_type = "original"
                elif '_blur.png' in filename:
                    sample_id_str = filename.replace('_blur.png', '')
                    augmentation_type = "blur"
                elif '_noise.png' in filename:
                    sample_id_str = filename.replace('_noise.png', '')
                    augmentation_type = "noise"
                elif '_mask.png' in filename:
                    sample_id_str = filename.replace('_mask.png', '')
                    augmentation_type = "mask"
                else:
                    continue
                
                if augmentation_type not in augmentation_types:
                    continue
                
                if sample_id_str not in sample_id_to_augmented_images:
                    sample_id_to_augmented_images[sample_id_str] = {}
                
                image_path = os.path.join(args.augmented_images_dir, filename)
                sample_id_to_augmented_images[sample_id_str][augmentation_type] = image_path
            
            print(f"{rank_info}Loaded augmented images for {len(sample_id_to_augmented_images)} samples")
            
            # Yield batches with augmented images
            for start in range(start_idx, end_idx, args.batch_size):
                batch_end = min(start + args.batch_size, end_idx)
                batch_indices = list(range(start, batch_end))
                
                # For each sample, create items for all augmentation types
                batch_items = []
                for idx in batch_indices:
                    original_item = dataset[idx]
                    sample_id = f"{original_item['id']}_{original_item['qid']}"
                    
                    # Create items for all augmentation types
                    for augmentation_type in augmentation_types:
                        if sample_id in sample_id_to_augmented_images:
                            if augmentation_type in sample_id_to_augmented_images[sample_id]:
                                # Use augmented image
                                augmented_item = original_item.copy()
                                augmented_item["image_path"] = sample_id_to_augmented_images[sample_id][augmentation_type]
                                augmented_item["augmentation_type"] = augmentation_type
                                augmented_item["sequence_id"] = 0
                                batch_items.append(augmented_item)
                        else:
                            # If sample not in augmented images, only yield original
                            if augmentation_type == "original":
                                augmented_item = original_item.copy()
                                augmented_item["augmentation_type"] = "original"
                                augmented_item["sequence_id"] = 0
                                batch_items.append(augmented_item)
                
                yield batch_items
        else:
            # Without augmentation, yield batches normally
            for start in range(start_idx, end_idx, args.batch_size):
                batch_end = min(start + args.batch_size, end_idx)
                batch_indices = list(range(start, batch_end))
                batch = [dataset[i] for i in batch_indices]
                
                # Add sequence_id and augmentation_type for compatibility
                for item in batch:
                    item["sequence_id"] = 0  # Will be set correctly by model generation
                    item["augmentation_type"] = "original"
                
                yield batch
    
    data_loader = distributed_dataloader()

    inference = ModelInference(
        base_model_path=args.base_model_path,
        processor_name=args.processor_name or args.base_model_path,
        data_loader=data_loader,
        max_new_tokens=args.max_new_tokens,
        split=args.split,
        peft_model_path=args.peft_model_path,
        num_return_sequences=args.num_return_sequences,
        on_reasoning=args.on_reasoning,
        accelerator=accelerator,
        with_augmentation=args.with_augmentation,
    )
    
    inference.run_inference(output_path)


if __name__ == "__main__":
    main()