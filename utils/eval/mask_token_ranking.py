"""
Mask-token reward ranking metric for preference datasets such as
``preference_dataset/HuatuoGPT-Vision-7B/slake/greedy/rrpo.json``.

For each entry in the dataset we run a single teacher-forced forward pass over
the (image, prompt, chosen) tuple through both a base model and a PEFT model
loaded on top of it (same setup as ``utils/reward/peft_reward_generator.py``:
the PEFT-wrapped model is loaded once and ``disable_adapter`` is used to obtain
base log-probabilities, so the tokenization and image features are identical
between the two branches). The per-token reward is the standard log ratio

    r_i = log p_PEFT(t_i | t_{<i}) - log p_BASE(t_i | t_{<i})

The dataset's ``chosen`` text marks "important" content using literal
``<mask>...</mask>`` tags (the tags themselves are stripped before tokenization
so they never enter the model input). We recover the character spans of the
masked content in the cleaned text and map them to completion-token indices
via the fast tokenizer's ``return_offsets_mapping``.

For each sample we then compute, for every percentile ``p ∈ --percentiles``
(default 5, 10, 50):

- ``any_in_top_p_pct``: ``1`` if at least one masked token sits in the top
  ``p%`` of completion tokens ranked by reward (descending), else ``0``.
- ``frac_masked_in_top_p_pct``: fraction of the sample's masked tokens that
  fall in that same top ``p%``.

The dataset-level numbers reported in the summary are the means of the
per-sample numbers above (averaged only over samples that actually contain at
least one masked token). We also report a pure global view where all
completion tokens from the whole dataset are pooled, ranked, and the masked
fraction of the resulting top ``p%`` is reported (``global_frac_masked_in_top_p_pct``).

Outputs:

- ``--csv_path`` (default ``./experiments/mask_token_ranking/<name>__<dataset>.csv``)
  with one row per evaluated sample.
- ``--summary_json`` (default alongside the CSV) with the dataset-level metrics.

From the repository root: ``source dpo_env/bin/activate``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from templates.conversation_templates import conversation_templates  # noqa: E402
from trl.data_utils import apply_chat_template  # noqa: E402

DEFAULT_BASE_MODEL_PATH = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
DEFAULT_PROCESSOR_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PERCENTILES: Tuple[int, ...] = (5, 10, 25, 50, 75)

MASK_OPEN = "<mask>"
MASK_CLOSE = "</mask>"
_MASK_RE = re.compile(re.escape(MASK_OPEN) + r"(.*?)" + re.escape(MASK_CLOSE), re.DOTALL)


def strip_mask_tags(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Remove ``<mask>...</mask>`` markers; return cleaned text and char spans.

    Each returned span ``(start, end)`` is a half-open interval over
    ``cleaned_text`` covering the content that was originally inside a
    ``<mask>...</mask>`` block.
    """
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    last_end = 0
    for m in _MASK_RE.finditer(text):
        prefix = text[last_end : m.start()]
        parts.append(prefix)
        cursor += len(prefix)
        content = m.group(1)
        parts.append(content)
        spans.append((cursor, cursor + len(content)))
        cursor += len(content)
        last_end = m.end()
    parts.append(text[last_end:])
    return "".join(parts), spans


def find_masked_token_indices(
    offsets: Sequence[Tuple[int, int]],
    mask_spans: Sequence[Tuple[int, int]],
    char_upper_bound: Optional[int] = None,
) -> List[int]:
    """Indices into ``offsets`` whose char range overlaps any of ``mask_spans``.

    Tokens with zero-width offsets (special tokens / EOS marker) are skipped.
    Tokens that start at or after ``char_upper_bound`` are also skipped (used
    to keep chat-template suffix tokens like ``<|im_end|>\\n`` out of the mask
    set).
    """
    if not mask_spans:
        return []
    out: List[int] = []
    for i, (s, e) in enumerate(offsets):
        if s is None or e is None or e <= s:
            continue
        if char_upper_bound is not None and s >= char_upper_bound:
            continue
        for ms, me in mask_spans:
            if max(s, ms) < min(e, me):
                out.append(i)
                break
    return out


def load_model_and_processor(
    base_model_path: str,
    peft_model_path: str,
    processor_name: str,
    device: torch.device,
    max_pixels: int = 512 * 28 * 28,
) -> Tuple[Any, Any]:
    """Load the base VLM, wrap it with the PEFT adapter, and load the processor."""
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        device_map={"": device},
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, peft_model_path, strict=False)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        processor_name,
        local_files_only=True,
        padding_side="left",
        max_pixels=max_pixels,
    )
    if not getattr(processor.tokenizer, "is_fast", False):
        raise RuntimeError(
            "A fast tokenizer is required for offset_mapping. "
            f"Got slow tokenizer for {processor_name!r}."
        )
    return model, processor


def _move_image_kwargs(
    processed: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Build the image-related kwargs (``pixel_values`` / ``image_grid_thw``) on ``device``."""
    out: Dict[str, Any] = {}
    pv = processed.get("pixel_values")
    if pv is not None:
        if isinstance(pv, torch.Tensor):
            out["pixel_values"] = pv.to(device)
        elif isinstance(pv, list):
            out["pixel_values"] = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in pv
            ]
        else:
            out["pixel_values"] = pv
    thw = processed.get("image_grid_thw")
    if thw is not None:
        if isinstance(thw, torch.Tensor):
            out["image_grid_thw"] = thw.to(device)
        else:
            out["image_grid_thw"] = thw
    return out


@torch.inference_mode()
def compute_per_token_rewards(
    model,
    processor,
    prompt_text: str,
    completion_text: str,
    image_path: str,
) -> Tuple[List[float], List[Tuple[int, int]], List[int]]:
    """
    Teacher-forced forward of ``prompt_text + completion_text`` through both
    the PEFT branch and the adapter-disabled base branch of ``model``.

    Returns
    -------
    rewards
        Per-completion-token log ratio ``log p_PEFT - log p_BASE``.
    offsets
        ``(start, end)`` char offsets into ``completion_text`` for each token.
        Tokens we add ourselves (the trailing EOS) get ``(-1, -1)``.
    completion_ids
        Token ids for the completion (last element is the appended EOS).
    """
    tokenizer = processor.tokenizer

    processed = processor(images=image_path, text=prompt_text, add_special_tokens=False)
    prompt_input_ids = processed["input_ids"][0]
    if isinstance(prompt_input_ids, torch.Tensor):
        prompt_ids_list = prompt_input_ids.tolist()
    else:
        prompt_ids_list = list(prompt_input_ids)

    completion_enc = tokenizer(
        completion_text, add_special_tokens=False, return_offsets_mapping=True
    )
    completion_ids: List[int] = list(completion_enc["input_ids"])
    offsets: List[Tuple[int, int]] = [tuple(o) for o in completion_enc["offset_mapping"]]
    completion_ids.append(int(tokenizer.eos_token_id))
    offsets.append((-1, -1))

    device = next(model.parameters()).device
    image_kwargs = _move_image_kwargs(processed, device)

    prompt_t = torch.tensor(prompt_ids_list, device=device, dtype=torch.long).unsqueeze(0)
    completion_t = torch.tensor(completion_ids, device=device, dtype=torch.long).unsqueeze(0)
    input_ids = torch.cat([prompt_t, completion_t], dim=1)
    n_prompt = int(prompt_t.shape[1])

    def _gather_completion_lps(logits_seq: torch.Tensor) -> List[float]:
        out: List[float] = []
        seq_len = int(logits_seq.shape[0])
        for i, tok_id in enumerate(completion_ids):
            pos = n_prompt + i - 1
            if pos < 0 or pos >= seq_len:
                out.append(float("nan"))
                continue
            lp = F.log_softmax(logits_seq[pos].float(), dim=-1)[int(tok_id)].item()
            out.append(lp)
        return out

    with model.disable_adapter():
        base_out = model(input_ids=input_ids, **image_kwargs)
        base_logits = base_out.logits[0]
    base_lps = _gather_completion_lps(base_logits)
    del base_out, base_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()

    peft_out = model(input_ids=input_ids, **image_kwargs)
    peft_logits = peft_out.logits[0]
    peft_lps = _gather_completion_lps(peft_logits)
    del peft_out, peft_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rewards = [p - b for p, b in zip(peft_lps, base_lps)]
    return rewards, offsets, completion_ids


def _topk_count(n: int, p_pct: int) -> int:
    """Number of tokens that constitute the top ``p_pct`` of ``n`` tokens.

    ``max(1, ceil(p/100 * n))``: a non-empty sample always has at least one
    token in the top X%, which matches the per-sample "is at least one masked
    token in the top X%" question.
    """
    if n <= 0:
        return 0
    return max(1, int(math.ceil((p_pct / 100.0) * n)))


def per_sample_metrics(
    rewards: Sequence[float],
    masked_token_indices: Sequence[int],
    percentiles: Sequence[int],
) -> Dict[str, Any]:
    """Per-sample ranking metrics; ``masked_token_indices`` are indices into ``rewards``."""
    n = len(rewards)
    masked_set = set(int(i) for i in masked_token_indices if 0 <= int(i) < n)
    n_masked = len(masked_set)
    out: Dict[str, Any] = {"n_tokens": n, "n_masked_tokens": n_masked}

    if n == 0 or n_masked == 0:
        for p in percentiles:
            out[f"any_in_top_{p}_pct"] = float("nan")
            out[f"frac_masked_in_top_{p}_pct"] = float("nan")
        return out

    # Sort indices by reward desc (stable on the index for tie-breaking).
    order = sorted(range(n), key=lambda i: (-rewards[i], i))
    for p in percentiles:
        k = _topk_count(n, int(p))
        top_set = set(order[:k])
        hits = len(top_set & masked_set)
        out[f"any_in_top_{p}_pct"] = 1.0 if hits > 0 else 0.0
        out[f"frac_masked_in_top_{p}_pct"] = hits / n_masked
    return out


def aggregate_dataset_metrics(
    per_sample: Sequence[Dict[str, Any]],
    global_rewards: Sequence[Tuple[float, bool]],
    percentiles: Sequence[int],
) -> Dict[str, Any]:
    """Average per-sample metrics + a pure global top-X% view across all tokens."""
    out: Dict[str, Any] = {
        "n_samples": len(per_sample),
        "n_samples_with_mask": sum(1 for s in per_sample if s["n_masked_tokens"] > 0),
        "total_tokens": int(sum(s["n_tokens"] for s in per_sample)),
        "total_masked_tokens": int(sum(s["n_masked_tokens"] for s in per_sample)),
    }
    valid = [s for s in per_sample if s["n_masked_tokens"] > 0 and s["n_tokens"] > 0]
    for p in percentiles:
        vals_any = [s[f"any_in_top_{p}_pct"] for s in valid]
        vals_frac = [s[f"frac_masked_in_top_{p}_pct"] for s in valid]
        out[f"any_in_top_{p}_pct"] = (
            float(sum(vals_any) / len(vals_any)) if vals_any else float("nan")
        )
        out[f"frac_masked_in_top_{p}_pct"] = (
            float(sum(vals_frac) / len(vals_frac)) if vals_frac else float("nan")
        )

        n_total = len(global_rewards)
        n_total_masked = sum(1 for _, m in global_rewards if m)
        if n_total == 0 or n_total_masked == 0:
            out[f"global_frac_masked_in_top_{p}_pct"] = float("nan")
            continue
        sorted_pool = sorted(global_rewards, key=lambda x: -x[0])
        k = _topk_count(n_total, int(p))
        top_masked = sum(1 for _, m in sorted_pool[:k] if m)
        out[f"global_frac_masked_in_top_{p}_pct"] = float(top_masked / n_total_masked)
    return out


def save_per_sample_csv(
    csv_path: str, records: Sequence[Dict[str, Any]], percentiles: Sequence[int]
) -> None:
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = ["index", "id", "qid", "image_path", "n_tokens", "n_masked_tokens"]
    for p in percentiles:
        fieldnames.append(f"any_in_top_{p}_pct")
        fieldnames.append(f"frac_masked_in_top_{p}_pct")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def save_summary_json(json_path: str, summary: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _default_csv_path(
    dataset_path: str, base_model_path: str, peft_model_path: Optional[str]
) -> str:
    """Mirror ``visual_grounding.py`` style: ``./experiments/mask_token_ranking/<name>__<stem>.csv``."""
    dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0] or "dataset"
    if peft_model_path:
        p = peft_model_path.rstrip(os.sep).rstrip("/")
        leaf = os.path.basename(p)
        ckpt = "checkpoint-"
        if leaf.startswith(ckpt) and leaf[len(ckpt):].isdigit():
            leaf = os.path.basename(os.path.dirname(p)) or leaf
        name = leaf or "peft"
    else:
        name = os.path.basename(base_model_path.rstrip(os.sep).rstrip("/")) or "model"
    return os.path.join(
        ".", "experiments", "mask_token_ranking", f"{name}__{dataset_stem}.csv"
    )


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Dataset JSON must be a list of records: {path}")
    return data


def run_pipeline(
    dataset_path: str,
    base_model_path: str,
    peft_model_path: str,
    processor_name: str,
    csv_path: str,
    summary_json_path: str,
    device: torch.device,
    max_samples: Optional[int] = None,
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
    log_every: int = 20,
) -> Dict[str, Any]:
    print(f"[mask_token_ranking] loading dataset: {dataset_path}", flush=True)
    raw = _load_dataset(dataset_path)
    total_in_file = len(raw)
    if max_samples is not None:
        raw = raw[: max(0, int(max_samples))]
    n = len(raw)
    print(
        f"[mask_token_ranking] {n} entries to evaluate "
        f"(file has {total_in_file})",
        flush=True,
    )

    print(
        f"[mask_token_ranking] loading model: base={base_model_path} "
        f"peft={peft_model_path}",
        flush=True,
    )
    model, processor = load_model_and_processor(
        base_model_path, peft_model_path, processor_name, device
    )
    qwen_template = conversation_templates.get_qwen()

    per_sample_records: List[Dict[str, Any]] = []
    global_rewards: List[Tuple[float, bool]] = []
    skipped = 0
    t0 = time.perf_counter()

    for idx, item in enumerate(raw):
        try:
            chosen_raw = item["chosen"]
            prompt = item["prompt"]
            image_path = item["image_path"]
        except KeyError as e:
            print(
                f"[mask_token_ranking] entry {idx}: missing required key {e}; skipping",
                flush=True,
            )
            skipped += 1
            continue

        if not os.path.isfile(image_path):
            print(
                f"[mask_token_ranking] entry {idx}: image not found at {image_path}; skipping",
                flush=True,
            )
            skipped += 1
            continue

        try:
            clean_chosen, mask_spans = strip_mask_tags(chosen_raw)

            sample = qwen_template.create_dpo_conversation(
                image_path=image_path,
                prompt=prompt,
                chosen=clean_chosen,
                rejected="",
            )
            applied = apply_chat_template(sample, processor)

            chosen_text = applied["chosen"]
            chosen_start = chosen_text.find(clean_chosen) if clean_chosen else 0
            if chosen_start < 0:
                print(
                    f"[mask_token_ranking] entry {idx}: clean chosen not found "
                    f"inside applied chat template; skipping",
                    flush=True,
                )
                skipped += 1
                continue
            shifted_spans = [
                (s + chosen_start, e + chosen_start) for (s, e) in mask_spans
            ]
            content_upper_bound = chosen_start + len(clean_chosen)

            rewards, offsets, _ids = compute_per_token_rewards(
                model=model,
                processor=processor,
                prompt_text=applied["prompt"],
                completion_text=chosen_text,
                image_path=image_path,
            )

            valid_pairs = [
                (i, r) for i, r in enumerate(rewards) if not math.isnan(r)
            ]
            kept_indices = [i for i, _ in valid_pairs]
            kept_rewards = [r for _, r in valid_pairs]
            kept_offsets = [offsets[i] for i in kept_indices]

            masked_indices_in_kept = find_masked_token_indices(
                kept_offsets, shifted_spans, char_upper_bound=content_upper_bound
            )

            metrics = per_sample_metrics(
                kept_rewards, masked_indices_in_kept, percentiles
            )
            metrics.update(
                {
                    "index": idx,
                    "id": item.get("id"),
                    "qid": item.get("qid"),
                    "image_path": image_path,
                }
            )
            per_sample_records.append(metrics)

            masked_set = set(masked_indices_in_kept)
            for i, r in enumerate(kept_rewards):
                global_rewards.append((r, i in masked_set))
        except Exception as e:
            print(
                f"[mask_token_ranking] entry {idx}: error during evaluation: {e}",
                flush=True,
            )
            traceback.print_exc()
            skipped += 1
            continue

        processed_count = idx + 1
        if processed_count % log_every == 0 or processed_count == n:
            elapsed = time.perf_counter() - t0
            rate = processed_count / elapsed if elapsed > 0 else 0.0
            eta = (n - processed_count) / rate if rate > 0 else float("nan")
            print(
                f"[mask_token_ranking] {processed_count}/{n} | "
                f"kept {len(per_sample_records)} skipped {skipped} | "
                f"elapsed {elapsed:.1f}s | rate {rate:.2f} samp/s | "
                f"eta {eta:.1f}s",
                flush=True,
            )

    summary = aggregate_dataset_metrics(per_sample_records, global_rewards, percentiles)
    summary.update(
        {
            "dataset_path": dataset_path,
            "dataset_file_total_records": total_in_file,
            "evaluated_samples": len(per_sample_records),
            "skipped_samples": skipped,
            "base_model_path": base_model_path,
            "peft_model_path": peft_model_path,
            "processor_name": processor_name,
            "percentiles": [int(p) for p in percentiles],
        }
    )

    save_per_sample_csv(csv_path, per_sample_records, percentiles)
    save_summary_json(summary_json_path, summary)
    summary["csv_path"] = csv_path
    summary["summary_json_path"] = summary_json_path
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mask-token reward ranking metric: for each preference-dataset "
            "entry, check whether any token inside <mask>...</mask> falls in "
            "the top X% of completion tokens by PEFT-vs-base reward."
        )
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Preference dataset JSON (e.g. preference_dataset/.../rrpo.json).",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Base VLM weights directory.",
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        required=True,
        help="PEFT/LoRA adapter directory loaded on top of --base_model_path.",
    )
    parser.add_argument(
        "--processor_name",
        type=str,
        default=DEFAULT_PROCESSOR_NAME,
        help="Processor name for AutoProcessor (default: Qwen2.5-VL-7B-Instruct).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "Per-sample metric CSV. Defaults to "
            "./experiments/mask_token_ranking/<name>__<dataset_stem>.csv. "
            "PEFT uses the last path segment, or the parent if the leaf is "
            "checkpoint-<digits>; base-only uses the last segment of --base_model_path."
        ),
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Aggregated metric JSON path. Defaults to <csv_stem>.summary.json.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to evaluate (prefix of dataset).",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="5,10,50",
        help="Comma-separated list of top-X percentiles to evaluate.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Print a progress line every N processed samples (default: 20).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset_path):
        raise SystemExit(f"Dataset JSON not found: {args.dataset_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    csv_path = args.csv_path or _default_csv_path(
        args.dataset_path, args.base_model_path, args.peft_model_path
    )
    summary_json = args.summary_json or (os.path.splitext(csv_path)[0] + ".summary.json")
    percentiles = tuple(
        int(x.strip()) for x in args.percentiles.split(",") if x.strip()
    )
    if not percentiles:
        raise SystemExit("--percentiles must contain at least one integer.")

    summary = run_pipeline(
        dataset_path=args.dataset_path,
        base_model_path=args.base_model_path,
        peft_model_path=args.peft_model_path,
        processor_name=args.processor_name,
        csv_path=csv_path,
        summary_json_path=summary_json,
        device=device,
        max_samples=args.max_samples,
        percentiles=percentiles,
        log_every=max(1, int(args.log_every)),
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved per-sample CSV to {csv_path}")
    print(f"Saved summary JSON to {summary_json}")


if __name__ == "__main__":
    main()
