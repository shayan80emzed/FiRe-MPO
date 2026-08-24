#!/usr/bin/env python3
"""
Enrich RRPO JSON (Slake or VQA-RAD) with MedSAM3 text prompts via GPT-4o-mini.

For each row, looks up ground-truth question + answer in the chosen dataset (match ``id``
and ``qid``, same normalization as ``src/dataset/vqa_dataset.py``), then asks
gpt-4o-mini for a short segmentation-style phrase suitable for MedSAM3.

Writes a new JSON file; does not modify the input RRPO file.

Requires: ``OPENAI_API_KEY`` in ``.env`` at the repo root (or in the process
environment). Uses ``python-dotenv`` to load ``<repo>/.env`` before reading the key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
import openai

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.vqa_dataset import VQADataset

# Repo-root .env, then cwd (allows override when run from another directory)
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

CHAT_MAX_ATTEMPTS = 3
CHAT_RETRY_DELAY_SEC = 10

# Full user message per dataset; placeholders: {question}, {answer}
MEDSAM3_USER_PROMPT_SLAKE = """Your task is to generate the prompt to a medical segmentation model. The model expects short text prompts to ground on the image.

Question:
{question}

Answer:
{answer}

Considering the question and the answer, respond with exactly one line: only the text that should be passed to the segmentation model as its prompt.
If there is no reasonable target to segment for this Q&A (e.g. the answer is only a modality like "MRI" or "CT" or it depends on the whole image rather than a specific part), respond with exactly: none

Examples:
Question: What is the modality of the image?
Answer: MRI
-> none

Question: Does the picture contain liver?
Answer: Yes
-> "liver"

Question: Does the picture contain kidney?
Answer: No
-> none

Question: What is the largest organ in the picture?
Answer: Heart
-> "heart"

Question: Is the lung healthy?
Answer: Yes
-> "lung"

Question: What diseases are included in the picture?
Answer: None
-> none

Question: What is the disease in the picture?
Answer: Lung Cancer
-> "lung cancer"

Question: Where is/are the abnormality located?
Answer: Left lung
-> "left lung"

Question: Are there abnormalities in this image?
Answer: Yes
-> "abnormalities"

Question: What system does the organ located on the top of this image belong to?
Answer: Respiratory System
-> "respiratory system"

Question: Are there organs in the picture that can promote blood flow?
Answer: Yes
-> "organs that can promote blood flow"

Question: What is the effect of the center organ in this picture?
Answer: Promote blood flow
-> "center organ in this picture"
"""

MEDSAM3_USER_PROMPT_VQA_RAD = """Your task is to generate the prompt to a medical segmentation model. The model expects short text prompts to ground on the image.

Question:
{question}

Answer:
{answer}

Considering the question and the answer, respond with exactly one line: only the text that should be passed to the segmentation model as its prompt.
If there is no reasonable target to segment for this Q&A (e.g. the answer is only a modality like "MRI" or "CT" or it depends on the whole image rather than a specific part), respond with exactly: none

Examples:


Question: are regions of the brain infarcted?
Answer: yes
-> "brain infarction"

Question: are the lungs normal appearing?
Answer: no
-> "lungs"

Question: what abnormality is seen?
Answer: blind-ending loop of bowel arising from the cecum
-> "blind loop of bowel"

Question: what is the radiological description of the mass?
Answer: hyperintense
-> "hyperintense mass"

Question: which region of the brain is impacted?
Answer: anterior surface
-> "anterior surface of the brain"

Question: is there a pneumothorax?
Answer: No
-> "lungs"

Question: is there a chest tube placed?
Answer: yes
-> "chest tube"

Question: what is the gender of this patient?
Answer: man
-> none

Question: are there rib fractures present?
Answer: no
-> "rib"

Question: in what vascular territory is the lesion located?
Answer: right mca
-> "lesion in right mca"

Question: does this scan represent an abnormality?
Answer: no
-> none

Question: is this a normal image?
Answer: no
-> "abnormalities"

Question: is there a shift of midline structures?
Answer: no
-> "midline structures"

Question: is there an brain bleed?
Answer: no
-> "brain"

Question: is there a calcification on the upper left lobe?
Answer: no
-> "upper left lobe"

Question: is there anything wrong with the lungs?
Answer: no
-> "lungs"

Question: is the lesion causing significant brainstem herniation?
Answer: no
-> "brainstem"

Question: is there a pneumothorax?
Answer: no
-> "lungs"

"""

MEDSAM3_USER_PROMPT_BY_DATASET: Dict[str, str] = {
    "slake": MEDSAM3_USER_PROMPT_SLAKE,
    "vqa_rad": MEDSAM3_USER_PROMPT_VQA_RAD,
}


def _medsam3_prompt_user_message(dataset_name: str, question: str, gt_answer: str) -> str:
    t = MEDSAM3_USER_PROMPT_BY_DATASET[dataset_name]
    return t.replace("{question}", question).replace("{answer}", gt_answer)


def _build_gt_index(
    dataset_name: str,
    splits: Tuple[str, ...],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Map (id, qid) -> {question, answer} using ``VQADataset``."""
    index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for split in splits:
        ds = VQADataset(dataset_name, split, reasoning=False)
        for i in range(len(ds)):
            row = ds[i]
            # ids may be non-int (e.g., UUIDs) depending on preprocessing; normalize to str
            key = (str(row["id"]), str(row["qid"]))
            index[key] = {
                "question": row["question"],
                "answer": row["answer"],
            }
    return index


async def _chat_once(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    user_text: str,
) -> Optional[str]:
    last_err: Optional[BaseException] = None
    for attempt in range(CHAT_MAX_ATTEMPTS):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            print(f"OpenAI error (attempt {attempt + 1}/{CHAT_MAX_ATTEMPTS}): {e}")
            if attempt < CHAT_MAX_ATTEMPTS - 1:
                await asyncio.sleep(CHAT_RETRY_DELAY_SEC)
    print(f"Giving up after errors: {last_err}")
    return None


async def _enrich_batch(
    items: list[dict],
    index: Dict[Tuple[str, str], Dict[str, str]],
    *,
    dataset_name: str,
    client: openai.AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    skip_existing: bool,
) -> None:
    async def one(item: dict) -> None:
        if skip_existing and item.get("medsam3_prompt"):
            return
        key = (str(item.get("id")), str(item.get("qid")))
        gt = index.get(key)
        if gt is None:
            print(f"Warning: no {dataset_name} row for id={key[0]!r} qid={key[1]!r}")
            item["medsam3_prompt"] = ""
            item["medsam3_prompt_gt_missing"] = True
            return
        user_text = _medsam3_prompt_user_message(dataset_name, gt["question"], gt["answer"])
        async with semaphore:
            raw = await _chat_once(client, model=model, user_text=user_text)
        if raw is None:
            item["medsam3_prompt"] = ""
            item["medsam3_prompt_error"] = True
            return
        line = raw.splitlines()[0].strip()
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()
        low = line.lower()
        if low == "none" or low == "none.":
            item["medsam3_prompt"] = ""
        else:
            item["medsam3_prompt"] = line

    await asyncio.gather(*[one(x) for x in items])


async def _run_async(args: argparse.Namespace) -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            f"Missing OPENAI_API_KEY. Add it to {REPO_ROOT / '.env'} "
            "(e.g. OPENAI_API_KEY=sk-...) or export it in your environment."
        )

    input_path = Path(args.input).expanduser().resolve()
    with open(input_path, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    index = _build_gt_index(args.dataset, tuple(args.splits))
    client = openai.AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.max_concurrent)

    for start in range(0, len(data), args.batch_size):
        batch = data[start : start + args.batch_size]
        print(f"Batch {start // args.batch_size + 1} / {(len(data) + args.batch_size - 1) // args.batch_size}")
        await _enrich_batch(
            batch,
            index,
            dataset_name=args.dataset,
            client=client,
            model=args.model,
            semaphore=sem,
            skip_existing=args.skip_existing,
        )
        # break

    return data


def main() -> None:
    p = argparse.ArgumentParser(
        description="Add medsam3_prompt to RRPO JSON via GPT-4o-mini + VQADataset GT."
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["slake", "vqa_rad"],
        required=True,
        help="Which VQADataset to use for (id,qid) question/answer lookup",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Dataset JSON splits to load for (id,qid) lookup (e.g. train val test)",
    )
    p.add_argument(
        "--slake-splits",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-concurrent", type=int, default=16)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows that already have non-empty medsam3_prompt",
    )
    args = p.parse_args()
    if args.slake_splits is not None:
        args.splits = args.slake_splits
    greedy = REPO_ROOT / "preference_dataset" / args.dataset / "greedy"
    if args.input is None:
        args.input = str(greedy / "rrpo.json")
    if args.output is None:
        args.output = str(greedy / "rrpo_with_medsam3_prompt.json")

    out = asyncio.run(_run_async(args))
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} records to {out_path}")
    
    cnt = 0
    for item in out:
        if item["medsam3_prompt"] == "":
            cnt += 1
    print(f"Percentage of rows with empty medsam3_prompt: {cnt / len(out)}")


if __name__ == "__main__":
    main()
