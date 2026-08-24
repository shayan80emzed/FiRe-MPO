"""
VGMED attention-alignment metrics following the **Medical-MLLMs-Fail** protocol
(https://github.com/Guimeng-Leo-Liu/Medical-MLLMs-Fail/blob/main/measure_attention.py).

For each sample we run **two** teacher-forced forwards on (image, prompt):

- ``question``: the raw dataset question (no extra suffix; appending the
  ``"Answer the question using a single word or phrase."`` instruction from the
  upstream script puts our model out-of-distribution, so it is omitted).
- ``general``:  ``GENERAL_QUESTION`` (``"Write a general description of the
  image."``), used purely as a normalization baseline.

Each prompt is fed through Qwen2.5-VL's chat template with
``add_generation_prompt=True``. We set ``output_attentions=True`` and read the
self-attention row at **sequence position ``-1``** (the last token of the chat
template, immediately before generation would start) to all **image placeholder**
keys. Per layer, attention across heads is **averaged** (no per-head
renormalization), giving ``att_ℓ ∈ ℝ^{N}`` over image tokens.

The reported attention map per layer is the element-wise ratio

    A_ℓ = att_question_ℓ / att_general_ℓ

reshaped to the LLM patch grid ``(H, W)`` from ``image_grid_thw``.

Given a binary mask ``M`` of image patches that cover the dataset bounding
boxes (floor / ceil token indexing on the **resized** image; union of boxes), we
compute per layer (same formulas as ``measure_attention.py``):

- **AR**: ``Σ A M / ((Σ A / N²) ‖M‖_0)``
- **KL**: ``D_KL( M̂ ‖ Â )`` with ``Â = A/Σ A``, ``M̂ = M/Σ M``
- **JS**: ``½ D_KL(M̂‖R̂) + ½ D_KL(Â‖R̂)``, ``R̂ = ½(M̂ + Â)``

``flash_attention_2`` does not return weights; this script **forces eager
attention** during the metrics forwards (config override). Prefer loading with
``--attn_implementation eager`` for reliability.

Dataset rows come from ``VQADataset("vgmed", "test", subset=...)``:
``image_path``, ``question``, ``bbox`` as ``[x, y, w, h]`` in **original image
pixels** (Slake-derived JSONs). A copy of the source image with those boxes
drawn (red outline) is saved next to the metrics CSV as ``<csv_stem>_bbox.png``
**only when** ``--sample_id`` is set.

If ``--sample_id`` is omitted, the script aggregates **per-layer means** over
the dataset (no bbox overlay images). Inputs are built in **batches** via
``processor.apply_chat_template(..., padding=True)``.

Metrics are written to **CSV** (``--csv_path``) and consumed unchanged by
``utils/eval/plot_grounding.py``. If omitted, the path defaults to
``./experiments/visual_grounding/<subset>/<name>.csv`` (``<subset>`` is
``loc`` or ``att``; ``combined`` when ``--subset`` is omitted). For
``--base_model_path``, ``<name>`` is the last path segment. For
``--peft_model_path``, ``<name>`` is the last segment unless it is
``checkpoint-<digits>`` (HF save folder), in which case ``<name>`` is the
parent directory (e.g. ``.../huatuo-dpo-slake/checkpoint-611`` →
``huatuo-dpo-slake``).

From the repository root: ``source dpo_env/bin/activate``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataset.vqa_dataset import (  # noqa: E402
    DATASET_DIR,
    VGMED_ATT_JSON,
    VGMED_DIR,
    VGMED_LOC_JSON,
    VQADataset,
)

DEFAULT_HUATUOGPT_VISION_PATH = (
    "/home/emzed/projects/aip-dolatab6/shared/model_weights/HuatuoGPT-Vision-7B"
)

# General-prompt baseline from Medical-MLLMs-Fail / measure_attention.py. The
# upstream script also appends ``"Answer the question using a single word or
# phrase."`` to both prompts; we omit it (the question prompt should be the raw
# dataset question to stay in-distribution for our trained models).
GENERAL_QUESTION = "Write a general description of the image."

# Protocol tag written into the ``query_position`` CSV column (read by plot_grounding.py
# for display only).
QUERY_POSITION_TAG = "medical_mllms_fail"

# Columns written by ``save_grounding_metrics_csv`` (read by ``plot_grounding.py``).
GROUNDING_METRICS_CSV_COLUMNS = [
    "dataset_json",
    "mode",
    "sample_id",
    "query_position",
    "base_model_path",
    "peft_model_path",
    "dataset_file_total_records",
    "num_dataset_samples_loaded",
    "contributing_rows_layer0",
    "batch_size",
    "layer_index",
    "ar",
    "kl",
    "js",
]


def _csv_float_cell(v: float) -> str:
    if v != v:
        return ""
    return repr(float(v))


def _default_grounding_csv_path(
    base_model_path: str,
    peft_model_path: Optional[str],
    subset: Optional[str] = None,
) -> str:
    if peft_model_path:
        p = peft_model_path.rstrip(os.sep).rstrip("/")
        leaf = os.path.basename(p)
        ckpt_prefix = "checkpoint-"
        if (
            leaf.startswith(ckpt_prefix)
            and len(leaf) > len(ckpt_prefix)
            and leaf[len(ckpt_prefix) :].isdigit()
        ):
            leaf = os.path.basename(os.path.dirname(p)) or leaf
        name = leaf or "model"
    else:
        chosen = base_model_path.rstrip(os.sep).rstrip("/")
        name = os.path.basename(chosen) or "model"
    subset_dir = subset if subset else "combined"
    return os.path.join(
        ".", "experiments", "visual_grounding", subset_dir, f"{name}.csv"
    )


def save_grounding_metrics_csv(
    csv_path: str,
    *,
    dataset_json: str,
    mode: str,
    sample_id: Optional[int],
    base_model_path: str,
    peft_model_path: Optional[str],
    dataset_file_total_records: Optional[int],
    num_dataset_samples_loaded: Optional[int],
    contributing_rows_layer0: Optional[int],
    batch_size: Optional[int],
    ar_vals: List[float],
    kl_vals: List[float],
    js_vals: List[float],
) -> str:
    """Write one row per decoder layer. Returns ``csv_path``."""
    if len(ar_vals) != len(kl_vals) or len(ar_vals) != len(js_vals):
        raise ValueError("ar_vals, kl_vals, js_vals must have the same length.")
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sid = "" if sample_id is None else str(sample_id)
    dtot = "" if dataset_file_total_records is None else str(dataset_file_total_records)
    nload = "" if num_dataset_samples_loaded is None else str(num_dataset_samples_loaded)
    crow = "" if contributing_rows_layer0 is None else str(contributing_rows_layer0)
    bs = "" if batch_size is None else str(batch_size)
    peft_s = peft_model_path or ""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GROUNDING_METRICS_CSV_COLUMNS)
        w.writeheader()
        for ell, (ar, kl, js) in enumerate(zip(ar_vals, kl_vals, js_vals)):
            w.writerow(
                {
                    "dataset_json": dataset_json,
                    "mode": mode,
                    "sample_id": sid,
                    "query_position": QUERY_POSITION_TAG,
                    "base_model_path": base_model_path,
                    "peft_model_path": peft_s,
                    "dataset_file_total_records": dtot,
                    "num_dataset_samples_loaded": nload,
                    "contributing_rows_layer0": crow,
                    "batch_size": bs,
                    "layer_index": str(ell),
                    "ar": _csv_float_cell(ar),
                    "kl": _csv_float_cell(kl),
                    "js": _csv_float_cell(js),
                }
            )
    return csv_path


@contextmanager
def _eager_attention_context(model: torch.nn.Module):
    """Temporarily force eager attention so ``output_attentions`` returns weights."""
    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    saved: List[Tuple[Any, str]] = []
    cfgs = []
    if hasattr(root.config, "text_config") and hasattr(root.config.text_config, "_attn_implementation"):
        cfgs.append(root.config.text_config)
    if hasattr(root, "model") and hasattr(root.model, "language_model"):
        lc = root.model.language_model.config
        if hasattr(lc, "_attn_implementation"):
            cfgs.append(lc)
    for c in cfgs:
        saved.append((c, c._attn_implementation))
        c._attn_implementation = "eager"
    try:
        yield
    finally:
        for c, v in saved:
            c._attn_implementation = v


@dataclass
class SampleRecord:
    image_path: str
    question: str
    bbox: Any
    raw: Dict[str, Any]


def vgmed_dataset_csv_label(subset: Optional[str]) -> str:
    """Value for the CSV ``dataset_json`` column (path-like label for plots)."""
    root = os.path.join(DATASET_DIR, VGMED_DIR)
    if subset == "loc":
        return os.path.join(root, VGMED_LOC_JSON)
    if subset == "att":
        return os.path.join(root, VGMED_ATT_JSON)
    return os.path.join(root, "__combined_loc_att__.json")


def _row_to_sample_record(row: Dict[str, Any]) -> SampleRecord:
    bbox = row.get("bbox") or []
    ip = row["image_path"]
    q = row["question"]
    return SampleRecord(
        image_path=ip,
        question=q,
        bbox=bbox,
        raw={"image": ip, "question": q, "bbox": bbox},
    )


def load_vgmed_sample(subset: Optional[str], sample_id: int) -> SampleRecord:
    ds = VQADataset("vgmed", "test", subset=subset)
    if sample_id < 0 or sample_id >= len(ds):
        raise IndexError(f"sample_id {sample_id} out of range [0, {len(ds) - 1}]")
    return _row_to_sample_record(ds[sample_id])


def load_vgmed_dataset(
    subset: Optional[str], max_samples: Optional[int] = None
) -> Tuple[List[SampleRecord], int]:
    """Load via ``VQADataset`` (optional cap). Returns ``(records, total_len)``."""
    ds = VQADataset("vgmed", "test", subset=subset)
    total = len(ds)
    n_take = total if max_samples is None else min(max(0, max_samples), total)
    records = [_row_to_sample_record(ds[i]) for i in range(n_take)]
    return records, total


def load_qwen25_vl(
    base_model_path: str,
    device: torch.device,
    peft_model_path: Optional[str] = None,
    attn_implementation: str = "eager",
) -> Tuple[Any, AutoProcessor]:
    """Load the VLM + processor.

    The HuatuoGPT-Vision checkpoints are Qwen2.5-VL based but ship without a
    processor, so we keep their special-cased loading (``Qwen2_5_VL`` class +
    processor pulled from the upstream Qwen2.5-VL-7B-Instruct repo, both with
    ``local_files_only=True`` to match the existing on-disk cache). All other
    base models go through the ``Auto*`` API and pull the processor from the
    same checkpoint as the model.
    """

    is_huatuo = "huatuo" in base_model_path.lower()
    if is_huatuo:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            device_map=None,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation=attn_implementation,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            max_pixels=512 * 28 * 28,
            local_files_only=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            device_map=None,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation=attn_implementation,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            base_model_path,
            max_pixels=512 * 28 * 28,
            local_files_only=True,
        )
    if peft_model_path:
        model = PeftModel.from_pretrained(model, peft_model_path, strict=False)
    processor.tokenizer.padding_side = "left"
    return model, processor


def _build_messages(
    image_path: str, text: str, system_text: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Build a user-turn message with the image and the given text content."""
    messages: List[Dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
    )
    return messages


def build_question_messages(
    image_path: str, question: str, system_text: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Raw dataset question (no extra instruction suffix; see module docstring)."""
    return _build_messages(image_path, question, system_text)


def build_general_messages(
    image_path: str, system_text: Optional[str] = None
) -> List[Dict[str, Any]]:
    """``GENERAL_QUESTION`` only (general-description baseline for normalization)."""
    return _build_messages(image_path, GENERAL_QUESTION, system_text)


def prepare_batched_from_conversations(
    processor,
    conversations: List[List[Dict[str, Any]]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Chat-template + processor pack, ``add_generation_prompt=True`` (left padded)."""
    feats = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}


def _processed_hw_for_model(image_path: str, processor) -> Tuple[int, int, int, int]:
    """(H0, W0, Hr, Wr) using the same ``smart_resize`` rules as the loaded image processor."""
    ip = processor.image_processor
    patch = int(ip.patch_size)
    merge = int(ip.merge_size)
    factor = patch * merge
    sz = getattr(ip, "size", {}) or {}
    min_px = getattr(ip, "min_pixels", None)
    if min_px is None:
        min_px = sz.get("shortest_edge", 56 * 56)
    max_px = getattr(ip, "max_pixels", None)
    if max_px is None:
        max_px = sz.get("longest_edge", 28 * 28 * 1280)
    min_px = int(min_px)
    max_px = int(max_px)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        W0, H0 = im.size
    Hr, Wr = smart_resize(H0, W0, factor=factor, min_pixels=min_px, max_pixels=max_px)
    return H0, W0, int(Hr), int(Wr)


def bbox_tokens_mask(
    bbox_list: Any,
    H0: int,
    W0: int,
    Hr: int,
    Wr: int,
    gh: int,
    gw: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Binary mask ``M`` of shape ``(gh, gw)``: floor/ceil token-cover of the union of
    bounding boxes after resizing to ``(Hr, Wr)``. Bboxes are ``[x, y, w, h]`` in
    **original image pixels** (Slake-style; matches measure_attention.py's xywh
    path through ``get_tokens_covering_bbox`` + ``convert_bbox_after_resize``).
    """
    M = torch.zeros(gh, gw, device=device, dtype=torch.float32)
    if not bbox_list:
        return M
    sx = Wr / float(W0)
    sy = Hr / float(H0)
    token_w = Wr / float(gw)
    token_h = Hr / float(gh)
    for box in bbox_list:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        x1, y1 = x * sx, y * sy
        x2, y2 = (x + w) * sx, (y + h) * sy
        x_min_t = max(int(math.floor(x1 / token_w)), 0)
        y_min_t = max(int(math.floor(y1 / token_h)), 0)
        x_max_t = min(int(math.ceil(x2 / token_w)), gw)
        y_max_t = min(int(math.ceil(y2 / token_h)), gh)
        if x_max_t <= x_min_t or y_max_t <= y_min_t:
            continue
        M[y_min_t:y_max_t, x_min_t:x_max_t] = 1.0
    return M


def save_bbox_overlay_image(
    image_path: str,
    bbox_list: Any,
    out_path: str,
    outline_rgb: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 3,
) -> str:
    """Draw dataset ``[x, y, w, h]`` boxes (original-pixel coords) and save."""
    with Image.open(image_path) as im:
        im = im.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    W, H = im.size
    if bbox_list:
        for box in bbox_list:
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            x0 = max(0, int(round(x)))
            y0 = max(0, int(round(y)))
            x1 = min(W - 1, int(round(x + w)))
            y1 = min(H - 1, int(round(y + h)))
            if x1 <= x0 or y1 <= y0:
                continue
            for d in range(line_width):
                draw.rectangle(
                    [x0 - d, y0 - d, x1 + d, y1 + d],
                    outline=outline_rgb,
                )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    im.save(out_path)
    return out_path


def attention_ratio_ar(A: torch.Tensor, M: torch.Tensor, eps: float = 1e-12) -> float:
    """``Σ A M / ((Σ A / N²) ‖M‖_0)`` for a non-negative ``A``."""
    N2 = A.numel()
    sA = A.sum()
    sM = M.sum()
    if sM < eps or sA < eps:
        return float("nan")
    num = (A * M).sum()
    den = (sA / N2) * sM
    return float((num / den.clamp_min(eps)).item())


def kl_hat_m_to_hat_a(M: torch.Tensor, A: torch.Tensor, eps: float = 1e-12) -> float:
    """``D_KL( M̂ ‖ Â )`` on flattened patch distributions."""
    m = M.flatten().float()
    a = A.flatten().float()
    sm, sa = m.sum(), a.sum()
    if sm < eps or sa < eps:
        return float("nan")
    mh = m / sm
    ah = a / sa
    return float((mh * (mh.clamp_min(eps).log() - ah.clamp_min(eps).log())).sum().item())


def js_hat_m_to_hat_a(M: torch.Tensor, A: torch.Tensor, eps: float = 1e-12) -> float:
    """Symmetric JS between normalized ``M`` and ``A``."""
    m = M.flatten().float()
    a = A.flatten().float()
    sm, sa = m.sum(), a.sum()
    if sm < eps or sa < eps:
        return float("nan")
    mh = m / sm
    ah = a / sa
    r = 0.5 * (mh + ah)
    kl_m = (mh * (mh.clamp_min(eps).log() - r.clamp_min(eps).log())).sum()
    kl_a = (ah * (ah.clamp_min(eps).log() - r.clamp_min(eps).log())).sum()
    return float((0.5 * kl_m + 0.5 * kl_a).item())


def _llm_patch_grid_hw(model: torch.nn.Module, image_grid_thw_row: torch.Tensor) -> Tuple[int, int, int]:
    """Returns ``(t, gh, gw)`` LLM merge grid (matches rope / image token count)."""
    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    merge = int(root.config.vision_config.spatial_merge_size)
    t = int(image_grid_thw_row[0].item())
    h_raw = int(image_grid_thw_row[1].item())
    w_raw = int(image_grid_thw_row[2].item())
    gh = h_raw // merge
    gw = w_raw // merge
    return t, gh, gw


def _extract_per_sample_image_attention(
    attentions: Tuple[torch.Tensor, ...],
    input_ids: torch.Tensor,
    image_token_id: int,
    expected_img_counts: List[int],
) -> List[Optional[torch.Tensor]]:
    """
    For each batch row ``b``, return a CPU float32 tensor of shape ``(L, n_img_b)``
    holding ``attn[b, :, -1, img_positions]`` averaged over heads, per layer.
    Rows whose image-token count != ``expected_img_counts[b]`` get ``None``.
    """
    B = int(input_ids.shape[0])
    L = len(attentions)
    out: List[Optional[torch.Tensor]] = [None] * B
    for b in range(B):
        img_pos = torch.nonzero(input_ids[b] == image_token_id, as_tuple=False).squeeze(-1)
        if int(img_pos.numel()) != int(expected_img_counts[b]):
            continue
        idx = img_pos.long()
        rows: List[torch.Tensor] = []
        for ell in range(L):
            a = attentions[ell]
            if a is None:
                rows = []
                break
            sliced = a[b, :, -1, idx].float()  # (H, n_img_b)
            rows.append(sliced.mean(dim=0).detach().cpu())
        if not rows:
            continue
        out[b] = torch.stack(rows, dim=0)  # (L, n_img_b)
    return out


def _new_layer_accum(num_layers: int) -> Dict[str, Any]:
    z = [0.0] * num_layers
    c = [0] * num_layers
    return {
        "sum_ar": list(z),
        "sum_kl": list(z),
        "sum_js": list(z),
        "cnt_ar": list(c),
        "cnt_kl": list(c),
        "cnt_js": list(c),
        "sample_rows": 0,
    }


def _accum_add(state: Dict[str, Any], ell: int, ar: float, kl: float, js: float) -> None:
    if ar == ar:
        state["sum_ar"][ell] += ar
        state["cnt_ar"][ell] += 1
    if kl == kl:
        state["sum_kl"][ell] += kl
        state["cnt_kl"][ell] += 1
    if js == js:
        state["sum_js"][ell] += js
        state["cnt_js"][ell] += 1


def _layer_means_from_accum(state: Dict[str, Any]) -> Tuple[List[float], List[float], List[float]]:
    L = len(state["sum_ar"])
    ar_m: List[float] = []
    kl_m: List[float] = []
    js_m: List[float] = []
    for ell in range(L):
        ca, ck, cj = state["cnt_ar"][ell], state["cnt_kl"][ell], state["cnt_js"][ell]
        ar_m.append(state["sum_ar"][ell] / ca if ca else float("nan"))
        kl_m.append(state["sum_kl"][ell] / ck if ck else float("nan"))
        js_m.append(state["sum_js"][ell] / cj if cj else float("nan"))
    return ar_m, kl_m, js_m


def _attention_layers_forward(
    model: Qwen2_5_VLForConditionalGeneration,
    feats: Dict[str, torch.Tensor],
    image_token_id: int,
    expected_img_counts: List[int],
) -> List[Optional[torch.Tensor]]:
    """One forward; returns per-sample ``(L, n_img)`` attention vectors on CPU."""
    with _eager_attention_context(model):
        out = model(**feats, output_attentions=True, return_dict=True, use_cache=False)
    atts = out.attentions
    if atts is None or len(atts) == 0:
        raise RuntimeError(
            "No attentions returned. Use eager attention (not flash_attention_2)."
        )
    for ell, a in enumerate(atts):
        if a is None:
            raise RuntimeError(f"attentions[{ell}] is None (attention impl may not expose weights).")
    vecs = _extract_per_sample_image_attention(atts, feats["input_ids"], image_token_id, expected_img_counts)
    del out
    return vecs


def _per_layer_metrics_from_normalized_attention(
    A_normalized_LhW: torch.Tensor,
    M: torch.Tensor,
) -> List[Tuple[float, float, float]]:
    """Compute AR/KL/JS for each layer given ``A`` of shape ``(L, gh, gw)`` and mask ``M``."""
    L = int(A_normalized_LhW.shape[0])
    M_cpu = M.detach().cpu().float()
    metrics: List[Tuple[float, float, float]] = []
    for ell in range(L):
        A_ell = A_normalized_LhW[ell]
        metrics.append(
            (
                attention_ratio_ar(A_ell, M_cpu),
                kl_hat_m_to_hat_a(M_cpu, A_ell),
                js_hat_m_to_hat_a(M_cpu, A_ell),
            )
        )
    return metrics


@torch.inference_mode()
def accumulate_batch_attention_metrics_mmf(
    model: Qwen2_5_VLForConditionalGeneration,
    processor,
    batch_records: List[SampleRecord],
    device: torch.device,
    state: Dict[str, Any],
    system_prompt: Optional[str] = None,
    eps_ratio: float = 1e-8,
) -> None:
    """
    Medical-MLLMs-Fail batch step: two forwards (question + general), per-layer
    element-wise ratio, then AR/KL/JS vs the bbox-covered patch mask.
    """
    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    image_token_id = int(root.config.image_token_id)

    q_convs = [
        build_question_messages(r.image_path, r.question, system_text=system_prompt)
        for r in batch_records
    ]
    g_convs = [
        build_general_messages(r.image_path, system_text=system_prompt) for r in batch_records
    ]
    feats_q = prepare_batched_from_conversations(processor, q_convs, device)
    feats_g = prepare_batched_from_conversations(processor, g_convs, device)

    B = len(batch_records)
    if "image_grid_thw" not in feats_q or feats_q["image_grid_thw"] is None:
        raise RuntimeError("Processor output missing image_grid_thw (question forward).")
    if "image_grid_thw" not in feats_g or feats_g["image_grid_thw"] is None:
        raise RuntimeError("Processor output missing image_grid_thw (general forward).")
    thw_q = feats_q["image_grid_thw"]
    thw_g = feats_g["image_grid_thw"]
    if thw_q.dim() == 1:
        thw_q = thw_q.unsqueeze(0)
    if thw_g.dim() == 1:
        thw_g = thw_g.unsqueeze(0)

    grids: List[Optional[Tuple[int, int, int]]] = []
    expected_q: List[int] = []
    expected_g: List[int] = []
    masks: List[Optional[torch.Tensor]] = []
    for b in range(B):
        t_q, gh_q, gw_q = _llm_patch_grid_hw(model, thw_q[b])
        t_g, gh_g, gw_g = _llm_patch_grid_hw(model, thw_g[b])
        if (t_q, gh_q, gw_q) != (t_g, gh_g, gw_g):
            grids.append(None)
            expected_q.append(t_q * gh_q * gw_q)
            expected_g.append(t_g * gh_g * gw_g)
            masks.append(None)
            continue
        grids.append((t_q, gh_q, gw_q))
        n_img = t_q * gh_q * gw_q
        expected_q.append(n_img)
        expected_g.append(n_img)
        rec = batch_records[b]
        H0, W0, Hr, Wr = _processed_hw_for_model(rec.image_path, processor)
        M_b = bbox_tokens_mask(rec.bbox, H0, W0, Hr, Wr, gh_q, gw_q, device)
        masks.append(M_b if float(M_b.sum().item()) > 0.0 else None)

    q_vecs = _attention_layers_forward(model, feats_q, image_token_id, expected_q)
    del feats_q
    if device.type == "cuda":
        torch.cuda.empty_cache()
    g_vecs = _attention_layers_forward(model, feats_g, image_token_id, expected_g)
    del feats_g
    if device.type == "cuda":
        torch.cuda.empty_cache()

    L_seen: Optional[int] = None
    for b in range(B):
        if grids[b] is None or masks[b] is None:
            continue
        qv = q_vecs[b]
        gv = g_vecs[b]
        if qv is None or gv is None:
            continue
        if qv.shape != gv.shape:
            continue
        L = int(qv.shape[0])
        if L_seen is None:
            L_seen = L
            if len(state.get("sum_ar", [])) != L:
                state.clear()
                state.update(_new_layer_accum(L))
        elif L != L_seen:
            continue

        t_b, gh_b, gw_b = grids[b]
        A_norm = qv / gv.clamp_min(eps_ratio)  # (L, n_img)
        if t_b != 1:
            try:
                A_norm = A_norm.reshape(L, t_b, gh_b, gw_b).mean(dim=1)
            except RuntimeError:
                continue
        else:
            try:
                A_norm = A_norm.reshape(L, gh_b, gw_b)
            except RuntimeError:
                continue

        metrics = _per_layer_metrics_from_normalized_attention(A_norm, masks[b])
        contributed = False
        for ell, (ar, kl, js) in enumerate(metrics):
            _accum_add(state, ell, ar, kl, js)
            if ell == 0 and (ar == ar or kl == kl or js == js):
                contributed = True
        if contributed:
            state["sample_rows"] += 1


@torch.inference_mode()
def compute_attention_metrics_single_sample(
    model: Qwen2_5_VLForConditionalGeneration,
    processor,
    rec: SampleRecord,
    device: torch.device,
    system_prompt: Optional[str] = None,
    eps_ratio: float = 1e-8,
) -> Tuple[List[float], List[float], List[float], Tuple[int, int, int], Tuple[int, int]]:
    """Single-sample MMF-protocol metrics. Returns ``(ar, kl, js, (t,gh,gw), (Hr,Wr))``."""
    state: Dict[str, Any] = {}
    accumulate_batch_attention_metrics_mmf(
        model=model,
        processor=processor,
        batch_records=[rec],
        device=device,
        state=state,
        system_prompt=system_prompt,
        eps_ratio=eps_ratio,
    )
    if not state.get("sum_ar"):
        raise RuntimeError("Failed to compute metrics for sample (no contributing layers).")
    ar, kl, js = _layer_means_from_accum(state)

    # Re-run the (cheap) processor pack to surface (t, gh, gw) / (Hr, Wr) for callers
    # (only used for debug-style metadata; no model forward).
    q_convs = [build_question_messages(rec.image_path, rec.question, system_text=system_prompt)]
    feats_q = prepare_batched_from_conversations(processor, q_convs, device)
    thw = feats_q["image_grid_thw"]
    if thw.dim() == 1:
        thw = thw.unsqueeze(0)
    t, gh, gw = _llm_patch_grid_hw(model, thw[0])
    _H0, _W0, Hr, Wr = _processed_hw_for_model(rec.image_path, processor)
    return ar, kl, js, (t, gh, gw), (Hr, Wr)


def _run_dataset_mean(
    dataset_label: str,
    vgmed_subset: Optional[str],
    base_model_path: str,
    peft_model_path: Optional[str],
    device: torch.device,
    csv_path: str,
    attn_implementation: str,
    system_prompt: Optional[str],
    batch_size: int,
    max_dataset_samples: Optional[int],
) -> Dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    if attn_implementation == "flash_attention_2":
        warnings.warn(
            "flash_attention_2 does not return attention weights; loading with eager for weight access.",
            stacklevel=2,
        )
        load_impl = "eager"
    else:
        load_impl = attn_implementation

    records, total_in_file = load_vgmed_dataset(vgmed_subset, max_dataset_samples)
    n = len(records)
    if n == 0:
        raise ValueError("Dataset is empty (after optional max_dataset_samples cap).")

    model, processor = load_qwen25_vl(
        base_model_path, device, peft_model_path, attn_implementation=load_impl
    )

    state: Dict[str, Any] = {}
    num_batches = (n + batch_size - 1) // batch_size
    t0 = time.perf_counter()

    for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
        batch_recs = records[start : start + batch_size]
        accumulate_batch_attention_metrics_mmf(
            model=model,
            processor=processor,
            batch_records=batch_recs,
            device=device,
            state=state,
            system_prompt=system_prompt,
        )
        processed = start + len(batch_recs)
        elapsed = time.perf_counter() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0
        eta = (n - processed) / rate if rate > 0 else float("nan")
        pct = 100.0 * processed / n if n else 0.0
        print(
            f"[visual_grounding] batch {batch_idx}/{num_batches} | "
            f"{processed}/{n} samples ({pct:.1f}%) | "
            f"elapsed {elapsed:.1f}s | rate {rate:.2f} samples/s | "
            f"eta {eta:.1f}s",
            flush=True,
        )

    ar_vals, kl_vals, js_vals = _layer_means_from_accum(state)

    save_grounding_metrics_csv(
        csv_path,
        dataset_json=dataset_label,
        mode="dataset_mean",
        sample_id=None,
        base_model_path=base_model_path,
        peft_model_path=peft_model_path,
        dataset_file_total_records=total_in_file,
        num_dataset_samples_loaded=n,
        contributing_rows_layer0=int(state.get("sample_rows", 0)),
        batch_size=batch_size,
        ar_vals=ar_vals,
        kl_vals=kl_vals,
        js_vals=js_vals,
    )

    return {
        "mode": "dataset_mean",
        "dataset_json": dataset_label,
        "num_dataset_samples_loaded": n,
        "dataset_file_total_records": total_in_file,
        "max_dataset_samples": max_dataset_samples,
        "batch_size": batch_size,
        "csv_path": csv_path,
        "bbox_overlay_image_path": None,
        "num_decoder_layers": len(ar_vals),
        "base_model_path": base_model_path,
        "peft_model_path": peft_model_path,
        "query_position": QUERY_POSITION_TAG,
        "contributing_rows_layer0": state.get("sample_rows", 0),
        "ar": ar_vals,
        "kl": kl_vals,
        "js": js_vals,
    }


def run_pipeline(
    dataset_label: str,
    vgmed_subset: Optional[str],
    sample_id: Optional[int],
    base_model_path: str,
    peft_model_path: Optional[str],
    device: torch.device,
    csv_path: str,
    attn_implementation: str = "eager",
    system_prompt: Optional[str] = None,
    batch_size: int = 4,
    max_dataset_samples: Optional[int] = None,
) -> Dict[str, Any]:
    if sample_id is None:
        return _run_dataset_mean(
            dataset_label=dataset_label,
            vgmed_subset=vgmed_subset,
            base_model_path=base_model_path,
            peft_model_path=peft_model_path,
            device=device,
            csv_path=csv_path,
            attn_implementation=attn_implementation,
            system_prompt=system_prompt,
            batch_size=batch_size,
            max_dataset_samples=max_dataset_samples,
        )

    rec = load_vgmed_sample(vgmed_subset, sample_id)

    if attn_implementation == "flash_attention_2":
        warnings.warn(
            "flash_attention_2 does not return attention weights; loading with eager for weight access.",
            stacklevel=2,
        )
        load_impl = "eager"
    else:
        load_impl = attn_implementation

    model, processor = load_qwen25_vl(
        base_model_path, device, peft_model_path, attn_implementation=load_impl
    )

    ar_vals, kl_vals, js_vals, (t, gh, gw), (Hr, Wr) = compute_attention_metrics_single_sample(
        model=model,
        processor=processor,
        rec=rec,
        device=device,
        system_prompt=system_prompt,
    )

    save_grounding_metrics_csv(
        csv_path,
        dataset_json=dataset_label,
        mode="single_sample",
        sample_id=sample_id,
        base_model_path=base_model_path,
        peft_model_path=peft_model_path,
        dataset_file_total_records=None,
        num_dataset_samples_loaded=None,
        contributing_rows_layer0=None,
        batch_size=None,
        ar_vals=ar_vals,
        kl_vals=kl_vals,
        js_vals=js_vals,
    )

    fig_root, _fig_ext = os.path.splitext(csv_path)
    bbox_image_path = f"{fig_root}_bbox.png"
    save_bbox_overlay_image(rec.image_path, rec.bbox, bbox_image_path)

    return {
        "mode": "single_sample",
        "sample_id": sample_id,
        "image": rec.image_path,
        "question": rec.question,
        "bbox": rec.bbox,
        "csv_path": csv_path,
        "bbox_overlay_image_path": bbox_image_path,
        "num_decoder_layers": len(ar_vals),
        "llm_grid_hw": [gh, gw],
        "processed_hw": [Hr, Wr],
        "base_model_path": base_model_path,
        "peft_model_path": peft_model_path,
        "query_position": QUERY_POSITION_TAG,
        "ar": ar_vals,
        "kl": kl_vals,
        "js": js_vals,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Attention-map AR / KL / JS vs decoder layer following the "
            "Medical-MLLMs-Fail protocol (question vs general-prompt ratio)."
        )
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["loc", "att"],
        help="VGMED subset for VQADataset (loc or att). Omit to concatenate loc + att.",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="Index into the loaded VGMED list. If omitted, aggregate per-layer means over the whole dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "Batch size for dataset-mean mode. Two forwards are run per batch "
            "(question + general); halve compared to greedy inference if OOM."
        ),
    )
    parser.add_argument(
        "--max_dataset_samples",
        type=int,
        default=None,
        help="Optional cap on number of VGMED samples to load (prefix of the VQADataset order).",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=DEFAULT_HUATUOGPT_VISION_PATH,
        help="Base Qwen2.5-VL weights directory (default: HuatuoGPT-Vision-7B)",
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        default=None,
        help="Optional LoRA/PEFT adapter directory; loaded on top of --base_model_path.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "Per-layer AR / KL / JS (plot with utils/eval/plot_grounding.py). "
            "If omitted: ./experiments/visual_grounding/<subset>/<name>.csv "
            "(<subset> is loc/att, or 'combined' when --subset is omitted); "
            "PEFT uses the last path segment, or the parent if the leaf is "
            "checkpoint-<digits>; base-only uses the last segment of --base_model_path."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Use eager for attention weights (flash_attention_2 cannot return weights).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt (Medical-MLLMs-Fail uses none; leave empty for parity).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.attn_implementation == "flash_attention_2":
        args.attn_implementation = "sdpa"

    dataset_label = vgmed_dataset_csv_label(args.subset)
    csv_path = args.csv_path or _default_grounding_csv_path(
        args.base_model_path, args.peft_model_path, args.subset
    )
    summary = run_pipeline(
        dataset_label=dataset_label,
        vgmed_subset=args.subset,
        sample_id=args.sample_id,
        base_model_path=args.base_model_path,
        peft_model_path=args.peft_model_path,
        device=device,
        csv_path=csv_path,
        attn_implementation=args.attn_implementation,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        max_dataset_samples=args.max_dataset_samples,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved metrics CSV to {summary['csv_path']}")
    if summary.get("bbox_overlay_image_path"):
        print(f"Saved bbox overlay image to {summary['bbox_overlay_image_path']}")


if __name__ == "__main__":
    main()
