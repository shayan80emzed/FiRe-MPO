from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
from typing import Any, Union, Optional
import torch
from trl.trainer.utils import pad, pad_to_length
from trl.trainer.dpo_trainer import DPOTrainer
import torch.nn as nn
import torch.nn.functional as F
from trl.trainer.utils import selective_log_softmax, flush_left
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def _make_unsloth_safe_dpo_collator(model, processor, pad_token_id: int):
    """
    Wrap our DPO collator so Unsloth leaves it alone.
    UnslothDPOTrainer replaces any data_collator that is not an UnslothVisionDataCollator
    with TransformersDataCollatorForLanguageModeling (which expects input_ids and breaks DPO).
    This wrapper subclasses UnslothVisionDataCollator and delegates to QwenDataCollatorForPreference.
    """
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator

    class _DPOCollatorWrapper(UnslothVisionDataCollator):
        __slots__ = ("_inner",)

        def __init__(self, model, processor, pad_token_id: int):
            super().__init__(model=model, processor=processor, max_seq_length=getattr(model, "max_seq_length", 16384))
            self._inner = QwenDataCollatorForPreference(pad_token_id=pad_token_id)

        def __call__(self, features, return_tensors=None):
            """Use our DPO collation; avoid parent's __call__ which expects chat messages."""
            return self._inner(features, return_tensors=return_tensors)

        def torch_call(self, examples: list) -> dict[str, Any]:
            return self._inner.torch_call(examples)

    return _DPOCollatorWrapper(model, processor, pad_token_id)


@dataclass
class QwenDataCollatorForPreference(DataCollatorMixin):

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        pixel_values = torch.cat([torch.tensor(example["pixel_values"]) for example in examples], dim=0)
        image_grid_thw = torch.cat([torch.tensor(example["image_grid_thw"]) for example in examples], dim=0)

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        output["pixel_values"] = pixel_values
        output["image_grid_thw"] = image_grid_thw

        return output
    
    

class QwenDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def process_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        """
        Same as `tokenize_row` but for vision models. Please refer to `tokenize_row` for more information.
        """
        processor, tokenizer = processing_class, processing_class.tokenizer  # the processing class is a processor
        processed_features = processor(images=features["images"], text=features["prompt"], add_special_tokens=False)

        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features["pixel_values"]
        image_grid_thw = processed_features["image_grid_thw"]
        
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        return output

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
                "pixel_values",
                "image_grid_thw",
            ]
    
    
    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:

        concatenated_batch = {}

        concatenated_batch['prompt_input_ids'] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        concatenated_batch['prompt_attention_mask'] = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)


        concatenated_batch['pixel_values'] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
        concatenated_batch['image_grid_thw'] = torch.cat([batch["image_grid_thw"], batch["image_grid_thw"]], dim=0)

        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        concatenated_batch['completion_input_ids'] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )

        concatenated_batch['completion_attention_mask'] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return concatenated_batch
            
    
    
    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        
        assert not "pixel_values_videos" in batch, "pixel_values_videos is not supported for QwenDPOTrainer"

        num_examples = batch['prompt_input_ids'].shape[0]
        
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}

        # Add image values to model kwargs
        model_kwargs['pixel_values'] = concatenated_batch['pixel_values']
        model_kwargs['image_grid_thw'] = concatenated_batch['image_grid_thw']

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps.sum(-1)

        output = {}

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        return output