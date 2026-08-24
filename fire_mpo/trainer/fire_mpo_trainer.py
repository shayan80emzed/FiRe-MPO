from dataclasses import dataclass, field
from trl import DPOConfig
from typing import Any, Dict, List, Union, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl.trainer.utils import pad, cap_exp, flush_left, flush_right, selective_log_softmax, pad_to_length
from src.trainer.qwen_dpo_trainer import QwenDPOTrainer, QwenDataCollatorForPreference
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from torch import autocast
from contextlib import nullcontext

from fire_mpo.loss.fire_mpo import compute_bidirectional_kl, token_wise_kl


def _make_unsloth_safe_fire_mpo_collator(model, processor, pad_token_id: int):
    """
    Wrap our FiRe-MPO collator so Unsloth leaves it alone.
    Unsloth replaces any data_collator that is not an UnslothVisionDataCollator with a default one
    that expects input_ids (and breaks FiRe-MPO). This wrapper subclasses UnslothVisionDataCollator
    and delegates to DataCollatorForFiReMPO.
    """
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator

    class _RRPOCollatorWrapper(UnslothVisionDataCollator):
        __slots__ = ("_inner",)

        def __init__(self, model, processor, pad_token_id: int):
            super().__init__(model=model, processor=processor, max_seq_length=getattr(model, "max_seq_length", 16384))
            self._inner = DataCollatorForFiReMPO(pad_token_id=pad_token_id)

        def __call__(self, features, return_tensors=None):
            return self._inner(features, return_tensors=return_tensors)

        def torch_call(self, examples: list) -> dict[str, Any]:
            return self._inner.torch_call(examples)

    return _RRPOCollatorWrapper(model, processor, pad_token_id)


@dataclass
class FiReMPOConfig(DPOConfig):
    """
    Configuration for FiRe-MPO (paper Eq. 3–5 / Appendix B).

    Paper hyperparameters:
      alpha   (α) — bidirectional token-wise KL weight
      gamma   (γ) — visual dual-pair ranking weight: (v', y-) ≻ (v', y+)
      lambda_ (λ) — forward-KL share in α · (λ FKL + (1-λ) RKL)
      beta    (β) — preference temperature (inherited from DPOConfig)

    Legacy aliases (rrpo_alpha, rrpo_alpha_v3, tkl_share) remain for old CLIs.
    Ablation-only: alpha_v1 / alpha_v2 (not part of the default FiRe-MPO recipe).
    """
    alpha: Optional[float] = field(
        default=0.01,
        metadata={"help": "α: coefficient for bidirectional token-wise KL (Eq. 5)."},
    )
    gamma: Optional[float] = field(
        default=None,
        metadata={
            "help": "γ: weight for visual preference (v', y-) ≻ (v', y+) (Eq. 3). "
            "Paper default is 0.1. Only applied when > 0 and samples provide rejected_image_path."
        },
    )
    lambda_: float = field(
        default=0.5,
        metadata={
            "help": "λ: forward-KL share in α · (λ FKL(ref‖π) + (1-λ) RKL(π‖ref)) (Eq. 4)."
        },
    )
    alpha_v1: Optional[float] = field(
        default=None,
        metadata={
            "help": "Ablation: weight for (v, y+) ≻ (v', y+). Not used in default FiRe-MPO."
        },
    )
    alpha_v2: Optional[float] = field(
        default=None,
        metadata={
            "help": "Ablation: weight for (v, y+) ≻ (v', y-). Not used in default FiRe-MPO."
        },
    )
    # --- Legacy aliases (synced in __post_init__) ---
    rrpo_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Legacy alias for alpha."},
    )
    rrpo_alpha_v1: Optional[float] = field(
        default=None,
        metadata={"help": "Legacy alias for alpha_v1."},
    )
    rrpo_alpha_v2: Optional[float] = field(
        default=None,
        metadata={"help": "Legacy alias for alpha_v2."},
    )
    rrpo_alpha_v3: Optional[float] = field(
        default=None,
        metadata={"help": "Legacy alias for gamma."},
    )
    tkl_share: Optional[float] = field(
        default=None,
        metadata={"help": "Legacy alias for lambda_."},
    )

    def __post_init__(self):
        if self.rrpo_alpha is not None:
            self.alpha = self.rrpo_alpha
        else:
            self.rrpo_alpha = self.alpha

        if self.rrpo_alpha_v3 is not None:
            self.gamma = self.rrpo_alpha_v3
        else:
            self.rrpo_alpha_v3 = self.gamma

        if self.rrpo_alpha_v1 is not None:
            self.alpha_v1 = self.rrpo_alpha_v1
        else:
            self.rrpo_alpha_v1 = self.alpha_v1

        if self.rrpo_alpha_v2 is not None:
            self.alpha_v2 = self.rrpo_alpha_v2
        else:
            self.rrpo_alpha_v2 = self.alpha_v2

        if self.tkl_share is not None:
            self.lambda_ = float(self.tkl_share)
        else:
            self.tkl_share = float(self.lambda_)

        if hasattr(super(), "__post_init__"):
            super().__post_init__()


@dataclass
class DataCollatorForFiReMPO(QwenDataCollatorForPreference):
    """
    Same as the DataCollatorForPreference class in the trl library, but with the addition of the chosen_phrase_mask and rejected_phrase_mask.
    """

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        output = super().torch_call(examples)
        output["chosen_phrase_mask"] = [torch.tensor(example["chosen_phrase_mask"]) for example in examples]
        output["rejected_phrase_mask"] = [torch.tensor(example["rejected_phrase_mask"]) for example in examples]
        
        output["chosen_phrase_mask"] = pad(output["chosen_phrase_mask"], padding_value=0)
        output["rejected_phrase_mask"] = pad(output["rejected_phrase_mask"], padding_value=0)

        output["pixel_values_l"] = torch.cat([torch.tensor(example["pixel_values_l"]) for example in examples], dim=0)
        output["image_grid_thw_l"] = torch.cat([torch.tensor(example["image_grid_thw_l"]) for example in examples], dim=0)
        output["has_rejected_image"] = torch.tensor(
            [bool(example["has_rejected_image"]) for example in examples], dtype=torch.bool
        )

        return output
    

class FiReMPOTrainer(QwenDPOTrainer):
    """
    Trainer for FIRE-MPO (fine-grained regularized medical preference optimization).
    """
    def __init__(self, model, args: FiReMPOConfig, **kwargs):
        super().__init__(model, args=args, **kwargs)
        self.alpha = self.args.alpha
        self.rrpo_alpha = self.args.rrpo_alpha


    @staticmethod
    def create_phrase_mask(input_ids: list[int], mask_start_token_id: int, mask_end_token_id: int) -> list[int]:
        """
        Create a phrase mask that marks tokens between <mask> and </mask> as 1, others as 0.
        Handles multiple mask pairs in the sequence.
        """
        phrase_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i] == mask_start_token_id:
                # Find the corresponding end token
                j = i + 1
                while j < len(input_ids) and input_ids[j] != mask_end_token_id:
                    phrase_mask[j] = 1  # Mark tokens between start and end as 1
                    j += 1
                if j < len(input_ids):  # Found the end token
                    i = j + 1  # Skip past the end token
                else:
                    raise ValueError(f"No </mask> found after <mask> at position {i}")
            else:
                i += 1
        return phrase_mask
    
    @staticmethod
    def get_inputs_without_mask(input_ids: list[int], phrase_mask: list[int], mask_start_token_id: int, mask_end_token_id: int) -> tuple[list[int], list[int]]:
        """
        Remove all mask tokens from input_ids and corresponding positions from phrase_mask.
        """
        new_input_ids = []
        new_phrase_mask = []
        
        for i in range(len(input_ids)):
            if input_ids[i] not in [mask_start_token_id, mask_end_token_id]:
                new_input_ids.append(input_ids[i])
                new_phrase_mask.append(phrase_mask[i])
        
        return new_input_ids, new_phrase_mask

    
    @staticmethod
    def sanity_mask_check(input_ids: list[int], mask_start_token_id: int, mask_end_token_id: int):
        """
        Asserts that the number of mask start tokens equals the number of mask end tokens,
        and that after each mask start there is a mask end (no nesting or overlap).
        """
        mask_start_indices = [i for i, t in enumerate(input_ids) if t == mask_start_token_id]
        mask_end_indices = [i for i, t in enumerate(input_ids) if t == mask_end_token_id]
        assert len(mask_start_indices) == len(mask_end_indices), (
            f"Number of <mask> ({len(mask_start_indices)}) and </mask> ({len(mask_end_indices)}) tokens do not match."
        )
        # Check that after each mask start is a mask end, and no nesting
        last_end = -1
        for start_idx in mask_start_indices:
            # Find the next end token after this start
            try:
                end_idx = next(i for i in mask_end_indices if i > start_idx)
            except StopIteration:
                raise AssertionError(f"No </mask> found after <mask> at position {start_idx}")
            assert end_idx > start_idx, f"</mask> at {end_idx} does not come after <mask> at {start_idx}"
            assert start_idx > last_end, f"Nested <mask> found at {start_idx}"
            last_end = end_idx
            # Remove this end_idx from mask_end_indices to prevent reuse
            mask_end_indices.remove(end_idx)
        
        
        
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
        
        MASK_START_TOKEN_ID = tokenizer.convert_tokens_to_ids("<mask>")
        MASK_END_TOKEN_ID = tokenizer.convert_tokens_to_ids("</mask>")
        
        assert MASK_START_TOKEN_ID is not None
        assert MASK_END_TOKEN_ID is not None
        
        processed_features = processor(images=features["images"], text=features["prompt"], add_special_tokens=False)

        
        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features["pixel_values"]
        image_grid_thw = processed_features["image_grid_thw"]

        rejected_image_path = features.get("rejected_image_path")
        if rejected_image_path:
            processed_l = processor(
                images=[rejected_image_path], text=features["prompt"], add_special_tokens=False
            )
            pixel_values_l = processed_l["pixel_values"]
            image_grid_thw_l = processed_l["image_grid_thw"]
            has_rejected_image = True
        else:
            # Share storage with chosen-image tensors (no duplicate vision memory).
            pixel_values_l = pixel_values
            image_grid_thw_l = image_grid_thw
            has_rejected_image = False

        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]
        
        # FiReMPOTrainer.sanity_mask_check(chosen_input_ids, MASK_START_TOKEN_ID, MASK_END_TOKEN_ID)
        # FiReMPOTrainer.sanity_mask_check(rejected_input_ids, MASK_START_TOKEN_ID, MASK_END_TOKEN_ID)
        
        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]
        
        # Create phrase masks for chosen and rejected sequences
        phrase_mask_chosen = FiReMPOTrainer.create_phrase_mask(chosen_input_ids, MASK_START_TOKEN_ID, MASK_END_TOKEN_ID)
        phrase_mask_rejected = FiReMPOTrainer.create_phrase_mask(rejected_input_ids, MASK_START_TOKEN_ID, MASK_END_TOKEN_ID)
        
        # Remove mask tokens from input_ids and phrase masks
        chosen_input_ids, phrase_mask_chosen = FiReMPOTrainer.get_inputs_without_mask(chosen_input_ids, phrase_mask_chosen, MASK_START_TOKEN_ID, MASK_END_TOKEN_ID)
        rejected_input_ids, phrase_mask_rejected = FiReMPOTrainer.get_inputs_without_mask(rejected_input_ids, phrase_mask_rejected, MASK_START_TOKEN_ID, MASK_END_TOKEN_ID)        
        
        

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_l": pixel_values_l,
            "image_grid_thw_l": image_grid_thw_l,
            "has_rejected_image": has_rejected_image,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "chosen_phrase_mask": phrase_mask_chosen,
            "rejected_phrase_mask": phrase_mask_rejected,
        }

        return output

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "pixel_values",
                "image_grid_thw",
                "pixel_values_l",
                "image_grid_thw_l",
                "has_rejected_image",
                "chosen_input_ids",
                "rejected_input_ids",
                "chosen_phrase_mask",
                "rejected_phrase_mask",
            ]
    
    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        
        concatenated_batch = QwenDPOTrainer.concatenated_inputs(batch, padding_value)
        
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        concatenated_batch['phrase_attention_mask'] = torch.cat(
            (
                pad_to_length(batch["chosen_phrase_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_phrase_mask"], max_completion_length, pad_value=0),
            ),
        )
        
        return concatenated_batch

    @staticmethod
    def build_concatenated_batch_visual_pair(
        batch: dict[str, Union[list, torch.LongTensor]],
        padding_value: int,
        *,
        pixel_values_second: torch.Tensor,
        image_grid_second: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_phrase_mask: torch.Tensor,
        pixel_values_chosen: Optional[torch.Tensor] = None,
        image_grid_chosen: Optional[torch.Tensor] = None,
        chosen_input_ids: Optional[torch.Tensor] = None,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        chosen_phrase_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.LongTensor]:
        """
        Build a 2*B concatenated batch: chosen branch uses batch pixel_values (m_w) + chosen completion;
        rejected branch uses ``pixel_values_second`` (typically m_l) + the given completion/mask (phrase-level).

        Optional overrides (e.g. RRPO v3): use ``pixel_values_chosen`` / ``chosen_input_ids`` so both
        branches can share m_l with different completions.
        """
        pv0 = pixel_values_chosen if pixel_values_chosen is not None else batch["pixel_values"]
        ig0 = image_grid_chosen if image_grid_chosen is not None else batch["image_grid_thw"]
        c_ids = chosen_input_ids if chosen_input_ids is not None else batch["chosen_input_ids"]
        c_am = chosen_attention_mask if chosen_attention_mask is not None else batch["chosen_attention_mask"]
        c_pm = chosen_phrase_mask if chosen_phrase_mask is not None else batch["chosen_phrase_mask"]

        concatenated_batch: dict[str, torch.LongTensor] = {}
        concatenated_batch["prompt_input_ids"] = torch.cat(
            [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
        )
        concatenated_batch["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        concatenated_batch["pixel_values"] = torch.cat([pv0, pixel_values_second], dim=0)
        concatenated_batch["image_grid_thw"] = torch.cat([ig0, image_grid_second], dim=0)

        max_completion_length = max(c_ids.shape[1], rejected_input_ids.shape[1])
        concatenated_batch["completion_input_ids"] = torch.cat(
            (
                pad_to_length(c_ids, max_completion_length, pad_value=padding_value),
                pad_to_length(rejected_input_ids, max_completion_length, pad_value=padding_value),
            ),
            dim=0,
        )
        concatenated_batch["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(c_am, max_completion_length, pad_value=0),
                pad_to_length(rejected_attention_mask, max_completion_length, pad_value=0),
            ),
            dim=0,
        )
        concatenated_batch["phrase_attention_mask"] = torch.cat(
            (
                pad_to_length(c_pm, max_completion_length, pad_value=0),
                pad_to_length(rejected_phrase_mask, max_completion_length, pad_value=0),
            ),
            dim=0,
        )
        return concatenated_batch

    def _run_fire_mpo_forward(
        self,
        model: nn.Module,
        concatenated_batch: dict[str, torch.LongTensor],
        num_examples: int,
    ) -> dict[str, torch.Tensor]:
        model_kwargs: dict[str, Any] = {}
        model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        model_kwargs["image_grid_thw"] = concatenated_batch["image_grid_thw"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        phrase_attention_mask = concatenated_batch["phrase_attention_mask"]

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), phrase_attention_mask), dim=1
        )

        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps.sum(-1)

        output: dict[str, torch.Tensor] = {}
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()
        output["chosen_logits"] = logits[:num_examples, :-1]
        output["chosen_tkl_loss_mask"] = loss_mask[:num_examples, : logits[:num_examples, :-1].shape[1]]
        return output

    def _compute_ref_fire_mpo_forward(self, concatenated_batch: dict[str, torch.LongTensor], num_examples: int):
        compte_ref_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    return self._run_fire_mpo_forward(self.model, concatenated_batch, num_examples)
            return self._run_fire_mpo_forward(self.ref_model, concatenated_batch, num_examples)
    

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        
        assert not "pixel_values_videos" in batch, "pixel_values_videos is not supported for QwenDPOTrainer"

        num_examples = batch['prompt_input_ids'].shape[0]
        
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)
        return self._run_fire_mpo_forward(model, concatenated_batch, num_examples)
    
    def compute_ref_log_probs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        compte_ref_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
        return ref_model_output
    

    def compute_tkl_loss(
        self,
        chosen_logits: torch.Tensor,
        ref_chosen_logits: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward KL(ref ‖ policy) at phrase positions (Eq. 4 FKL term)."""
        return token_wise_kl(ref_chosen_logits, chosen_logits, loss_mask)
    
    def compute_nll_loss(self, logps: torch.Tensor) -> torch.Tensor:
        count_nonzero = (logps != 0).sum(dim=-1).clamp(min=1)
        per_sample_sum = -logps.sum(dim=-1) / count_nonzero
        return per_sample_sum.mean()
    
    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        _lv1 = _cr_v1 = _rr_v1 = None
        _lv2 = _cr_v2 = _rr_v2 = None
        _lv3 = _cr_v3 = _rr_v3 = None
        alpha_v1 = alpha_v2 = alpha_v3 = None
        w = None
        loss_base_rrpo: Optional[torch.Tensor] = None

        if self.args.use_liger_loss:
            raise ValueError("Liger loss is not supported for RRPO")
        else:
            model_output = self.concatenated_forward(model, batch)

            # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
            if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                raise ValueError("Ref log probs are not supported for RRPO, and sorry how???")
            else:
                ref_output = self.compute_ref_log_probs(batch)

            # RRPO uses the sigmoid (DPO) preference loss only
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                model_output["chosen_logps"],
                model_output["rejected_logps"],
                ref_output["chosen_logps"],
                ref_output["rejected_logps"],
                "sigmoid",
                model_output,
            )
            loss_base_rrpo = losses

            alpha_v1 = getattr(self.args, "alpha_v1", None)
            alpha_v2 = getattr(self.args, "alpha_v2", None)
            gamma = getattr(self.args, "gamma", None)
            # legacy fallbacks
            if alpha_v1 is None:
                alpha_v1 = getattr(self.args, "rrpo_alpha_v1", None)
            if alpha_v2 is None:
                alpha_v2 = getattr(self.args, "rrpo_alpha_v2", None)
            if gamma is None:
                gamma = getattr(self.args, "rrpo_alpha_v3", None)
            alpha_v3 = gamma
            use_v1 = alpha_v1 is not None and alpha_v1 > 0
            use_v2 = alpha_v2 is not None and alpha_v2 > 0
            use_v3 = alpha_v3 is not None and alpha_v3 > 0
            if use_v1 or use_v2 or use_v3:
                num_e = batch["prompt_input_ids"].shape[0]
                w = batch["has_rejected_image"].to(device=model_output["chosen_logps"].device).to(
                    dtype=model_output["chosen_logps"].dtype
                )
                if use_v1:
                    conc_v1 = self.build_concatenated_batch_visual_pair(
                        batch,
                        self.padding_value,
                        pixel_values_second=batch["pixel_values_l"],
                        image_grid_second=batch["image_grid_thw_l"],
                        rejected_input_ids=batch["chosen_input_ids"],
                        rejected_attention_mask=batch["chosen_attention_mask"],
                        rejected_phrase_mask=batch["chosen_phrase_mask"],
                    )
                    out_v1 = self._run_fire_mpo_forward(model, conc_v1, num_e)
                    ref_v1 = self._compute_ref_fire_mpo_forward(conc_v1, num_e)
                    _lv1, _cr_v1, _rr_v1 = self.dpo_loss(
                        out_v1["chosen_logps"],
                        out_v1["rejected_logps"],
                        ref_v1["chosen_logps"],
                        ref_v1["rejected_logps"],
                        "sigmoid",
                        model_output,
                    )
                    losses = losses + alpha_v1 * _lv1 * w
                if use_v2:
                    conc_v2 = self.build_concatenated_batch_visual_pair(
                        batch,
                        self.padding_value,
                        pixel_values_second=batch["pixel_values_l"],
                        image_grid_second=batch["image_grid_thw_l"],
                        rejected_input_ids=batch["rejected_input_ids"],
                        rejected_attention_mask=batch["rejected_attention_mask"],
                        rejected_phrase_mask=batch["rejected_phrase_mask"],
                    )
                    out_v2 = self._run_fire_mpo_forward(model, conc_v2, num_e)
                    ref_v2 = self._compute_ref_fire_mpo_forward(conc_v2, num_e)
                    _lv2, _cr_v2, _rr_v2 = self.dpo_loss(
                        out_v2["chosen_logps"],
                        out_v2["rejected_logps"],
                        ref_v2["chosen_logps"],
                        ref_v2["rejected_logps"],
                        "sigmoid",
                        model_output,
                    )
                    losses = losses + alpha_v2 * _lv2 * w
                if use_v3:
                    conc_v3 = self.build_concatenated_batch_visual_pair(
                        batch,
                        self.padding_value,
                        pixel_values_chosen=batch["pixel_values_l"],
                        image_grid_chosen=batch["image_grid_thw_l"],
                        chosen_input_ids=batch["rejected_input_ids"],
                        chosen_attention_mask=batch["rejected_attention_mask"],
                        chosen_phrase_mask=batch["rejected_phrase_mask"],
                        pixel_values_second=batch["pixel_values_l"],
                        image_grid_second=batch["image_grid_thw_l"],
                        rejected_input_ids=batch["chosen_input_ids"],
                        rejected_attention_mask=batch["chosen_attention_mask"],
                        rejected_phrase_mask=batch["chosen_phrase_mask"],
                    )
                    out_v3 = self._run_fire_mpo_forward(model, conc_v3, num_e)
                    ref_v3 = self._compute_ref_fire_mpo_forward(conc_v3, num_e)
                    _lv3, _cr_v3, _rr_v3 = self.dpo_loss(
                        out_v3["chosen_logps"],
                        out_v3["rejected_logps"],
                        ref_v3["chosen_logps"],
                        ref_v3["rejected_logps"],
                        "sigmoid",
                        model_output,
                    )
                    losses = losses + alpha_v3 * _lv3 * w

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]
            
        if (getattr(self.args, "alpha", None) or 0) > 0 or (
            self.args.rrpo_alpha is not None and self.args.rrpo_alpha > 0
        ):
            alpha = float(getattr(self.args, "alpha", None) or self.args.rrpo_alpha or 0)
            if alpha > 0:
                lambda_ = float(
                    getattr(self.args, "lambda_", None)
                    if getattr(self.args, "lambda_", None) is not None
                    else getattr(self.args, "tkl_share", 0.5) or 0.5
                )
                tkl_loss, rtkl_loss, mixed_tkl_loss = compute_bidirectional_kl(
                    model_output["chosen_logits"],
                    ref_output["chosen_logits"],
                    model_output["chosen_tkl_loss_mask"],
                    ref_output["chosen_tkl_loss_mask"],
                    lambda_=lambda_,
                )
                losses = losses + alpha * mixed_tkl_loss

        prefix = "eval_" if train_eval == "eval" else ""

        metrics[f"{prefix}loss/fire_mpo_base"] = self.accelerator.gather_for_metrics(loss_base_rrpo.detach()).mean().item()
        metrics[f"{prefix}loss/rrpo_base"] = metrics[f"{prefix}loss/fire_mpo_base"]
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()

        if _lv1 is not None:
            metrics[f"{prefix}loss/fire_mpo_v1"] = self.accelerator.gather_for_metrics(_lv1.detach()).mean().item()
            metrics[f"{prefix}loss/rrpo_v1"] = metrics[f"{prefix}loss/fire_mpo_v1"]
            metrics[f"{prefix}rewards/v1/chosen"] = self.accelerator.gather_for_metrics(_cr_v1).mean().item()
            metrics[f"{prefix}rewards/v1/rejected"] = self.accelerator.gather_for_metrics(_rr_v1).mean().item()
            metrics[f"{prefix}rewards/v1/accuracies"] = self.accelerator.gather_for_metrics((_cr_v1 > _rr_v1).float()).mean().item()

        if _lv2 is not None:
            metrics[f"{prefix}loss/fire_mpo_v2"] = self.accelerator.gather_for_metrics(_lv2.detach()).mean().item()
            metrics[f"{prefix}loss/rrpo_v2"] = metrics[f"{prefix}loss/fire_mpo_v2"]
            metrics[f"{prefix}rewards/v2/chosen"] = self.accelerator.gather_for_metrics(_cr_v2).mean().item()
            metrics[f"{prefix}rewards/v2/rejected"] = self.accelerator.gather_for_metrics(_rr_v2).mean().item()
            metrics[f"{prefix}rewards/v2/accuracies"] = self.accelerator.gather_for_metrics((_cr_v2 > _rr_v2).float()).mean().item()

        if _lv3 is not None:
            metrics[f"{prefix}loss/fire_mpo_gamma"] = self.accelerator.gather_for_metrics(_lv3.detach()).mean().item()
            metrics[f"{prefix}loss/rrpo_v3"] = metrics[f"{prefix}loss/fire_mpo_gamma"]
            metrics[f"{prefix}rewards/v3/chosen"] = self.accelerator.gather_for_metrics(_cr_v3).mean().item()
            metrics[f"{prefix}rewards/v3/rejected"] = self.accelerator.gather_for_metrics(_rr_v3).mean().item()
            metrics[f"{prefix}rewards/v3/accuracies"] = self.accelerator.gather_for_metrics((_cr_v3 > _rr_v3).float()).mean().item()

        metrics[f"{prefix}logps/chosen"] = self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        metrics[f"{prefix}logps/rejected"] = self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        metrics[f"{prefix}logits/chosen"] = self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        metrics[f"{prefix}logits/rejected"] = self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            metrics[f"{prefix}nll_loss"] = self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
        if (getattr(self.args, "alpha", None) or 0) > 0 or (
            self.args.rrpo_alpha is not None and self.args.rrpo_alpha > 0
        ):
            metrics[f"{prefix}tkl_loss"] = self.accelerator.gather_for_metrics(tkl_loss).detach().mean().item()
            metrics[f"{prefix}rtkl_loss"] = self.accelerator.gather_for_metrics(rtkl_loss).detach().mean().item()
            metrics[f"{prefix}mixed_tkl_loss"] = self.accelerator.gather_for_metrics(mixed_tkl_loss).detach().mean().item()
        return losses.mean(), metrics
