import sys
import os
# Add project root to Python path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig
from trl import SFTTrainer
from peft import LoraConfig
from src.dataset.sft_dataset import SFTDataset
from utils.train import find_target_linear_names

from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling



def main():
    
    # Load the model and processor
    model = AutoModelForImageTextToText.from_pretrained(
        "/model-weights/Qwen2.5-VL-7B-Instruct",
        local_files_only=True,
        dtype="bfloat16",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=True, padding_side="left", max_pixels=28 * 28 * 576)

    # Load the Medical VQA DPO dataset
    print("Loading Medical VQA SFT dataset...")
    dataset = SFTDataset(
        data_path="data/path_vqa/dpo_dataset.json", 
        # image_folder="/home/emzed/projects/aip-dolatab6/shared/datasets/vqa_rad",
        # max_samples=2,
    )
    
    # Lora Config (181M trainable params)
    lora_cfg = LoraConfig(
        target_modules=find_target_linear_names(model),
        r=128,
        lora_alpha=256,
        task_type="CAUSAL_LM"
    )

    # Train the model
    training_args = SFTConfig(
        output_dir="models/qwen25vl-sft-path_vqa",
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        num_train_epochs=1,
        logging_steps=10,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",  # Better learning rate schedule
        dataset_num_proc=8,
        dataloader_num_workers=8,
        report_to="wandb",
        run_name="qwen25vl-sft-path_vqa",
        # max_prompt_length=None,
        # max_completion_length=None,
        max_length=None,
    )
    
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_cfg,
        data_collator=DataCollatorForVisionLanguageModeling(processor),
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
