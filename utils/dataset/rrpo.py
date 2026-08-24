import json
import os
import re
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

import openai

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class RRPOPromptConfig:
    """VQA RRPO: locate uses {question}, {answer}, {output}; mask uses {sentence_a}, {sentence_b}."""

    locate_template: str
    mask_template: str


LOCATE_PROMPT_VQA = """You are evaluating a model's answer to a medical question.

Question: {question}
Correct answer (ground truth): {answer}
Model's full answer: {output}

Perform these four steps and respond with a single JSON object only (no other text):

1. **answer_sentence**: The minimal phrase or clause from the model's answer that states the final answer, using the model's exact wording. Preserve the model's language so the final answer is visible in context. Do not include extra clauses that follow. The answer sentence can end with a period or comma.
2. **final_answer**: Locate and extract the final answer to the question from the model's answer. Use as few words as possible (the first occurrence of the conclusive answer).
3. **is_correct**: true if the final answer is correct given the ground truth, false otherwise.
4. **opposite_answer**: Modify the answer sentence as little as possible to make a medically plausible alternative for the question.

Respond only with valid JSON in this exact shape:
{{"answer_sentence": "...", "final_answer": "...", "is_correct": true or false, "opposite_answer": "..."}}
"""

MASK_PROMPT_VQA = """You are given two sentences that are the same except for some phrases. Your task is to identify the phrases in each sentence that differs from the other.

Sentence A: {sentence_a}
Sentence B: {sentence_b}

Wrap only the differing phrases in <mask> </mask> tags (space after <mask> and before </mask>). Keep the rest of the sentence unchanged. Notice that the "Yes" and "No" are phrases that should be wrapped in <mask> </mask> tags. The differing phrases should be as short as possible.

Respond with a single JSON object only (no other text), with two keys:
- "sentence_a_masked": Sentence A with the differing phrase wrapped in <mask> </mask>
- "sentence_b_masked": Sentence B with the differing phrase wrapped in <mask> </mask>

Examples:

Sentence A: This image shows the abdomen.
Sentence B: This image shows the lung tissue.
sentence_a_masked: This image shows the <mask>abdomen</mask>.
sentence_b_masked: This image shows the <mask>lung tissue</mask>.

Sentence A: Yes, the image does show the liver.
Sentence B: No, the image does not show the liver.
sentence_a_masked: <mask>Yes</mask>, the image does <mask>show</mask> the liver.
sentence_b_masked: <mask>No</mask>, the image does <mask>not show</mask> the liver.

Sentence A: Yes, the image does contain a section of liver tissue.
Sentence B: No, the image does not contain a section of liver tissue.
sentence_a_masked: <mask>Yes</mask>, the image does <mask>contain</mask> a section of liver tissue.
sentence_b_masked: <mask>No</mask>, the image does <mask>not contain</mask> a section of liver tissue.

Sentence A: No, the image does not contain a kidney.
Sentence B: Yes, the image contains a kidney.
sentence_a_masked: <mask>No</mask>, the image <mask>does not contain</mask> a kidney.
sentence_b_masked: <mask>Yes</mask>, the image <mask>contains</mask> a kidney.
"""

DEFAULT_RRPO_PROMPT_CONFIG = RRPOPromptConfig(
    locate_template=LOCATE_PROMPT_VQA,
    mask_template=MASK_PROMPT_VQA,
)


def get_rrpo_prompt_config() -> RRPOPromptConfig:
    return DEFAULT_RRPO_PROMPT_CONFIG


def preprocess_rrpo_dataset_answers(train_data: list[dict]) -> None:
    """Remove placeholder token XXXX from ground-truth answers (mutates rows in place)."""
    for row in train_data:
        a = row.get("answer")
        if not isinstance(a, str) or not a:
            continue
        row["answer"] = a.replace("XXXX", "")


def _evaluation_model() -> str:
    return os.getenv("EVALUATION_MODEL_VQA", "gpt-4o-mini")


api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")

async_client = openai.AsyncOpenAI(api_key=api_key)
batch_size = 50
# If True: run first batch only; collect locate/mask prompts and GPT outputs, print once at end of main().
is_debug = False

CHAT_MAX_ATTEMPTS = 3
CHAT_RETRY_DELAY_SEC = 10


async def _chat_completion_create(**kwargs):
    """Call chat.completions.create; on failure wait and retry. Return None after final failure."""
    last_error: Optional[BaseException] = None
    for attempt in range(CHAT_MAX_ATTEMPTS):
        try:
            return await async_client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            print(f"OpenAI request failed (attempt {attempt + 1}/{CHAT_MAX_ATTEMPTS}): {e}")
            if attempt < CHAT_MAX_ATTEMPTS - 1:
                await asyncio.sleep(CHAT_RETRY_DELAY_SEC)
    print(f"Skipping request after {CHAT_MAX_ATTEMPTS} failed attempts: {last_error}")
    return None


def _debug_log_append(
    debug_log: list[dict],
    *,
    batch: int,
    sample_id: object,
    step: int,
    label: str,
    content: str,
) -> None:
    """Append one debug record (async-safe: single list append under GIL)."""
    debug_log.append(
        {
            "batch": batch,
            "sample_id": sample_id,
            "step": step,
            "label": label,
            "content": content,
        }
    )


def _print_debug_log(debug_log: list[dict]) -> None:
    """Print collected debug records in stable order (batch, sample, step)."""
    if not debug_log:
        print("Debug log: (empty)")
        return
    key = lambda r: (r["batch"], str(r["sample_id"]), r["step"])
    for r in sorted(debug_log, key=key):
        print("=" * 80)
        print(f"[batch {r['batch']} id={r['sample_id']!r}] {r['label']}")
        print("-" * 80)
        print(r["content"])
    print("=" * 80)
    print(f"Debug log: {len(debug_log)} record(s).")


def _normalize_chosen_rejected(s: str) -> str:
    """Postprocess chosen/rejected sentence: capital, <mask>/</mask> spacing, period end."""
    if not s or not s.strip():
        return s
    s = s.strip()
    # Exactly one space before <mask>, nothing (no space) after <mask>
    s = re.sub(r"\s+<mask>", " <mask>", s)
    s = re.sub(r"<mask>\s+", "<mask>", s)
    # Nothing before </mask>, exactly one space after </mask>
    s = re.sub(r"\s+</mask>", "</mask>", s)
    s = re.sub(r"</mask>\s+", "</mask> ", s)
    # Sentence starts with a capital letter
    s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
    # Sentence ends with a period (not a comma)
    s = s.rstrip()
    if s.endswith(","):
        s = s[:-1] + "."
    elif s and s[-1] not in ".!?":
        s = s + "."
    return s


def _parse_single_api_response(content: str, item: dict) -> dict:
    """Parse GPT response; fallback to safe defaults on failure."""
    try:
        # Try to extract JSON (handle markdown code blocks)
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        final_answer = data.get("final_answer", "").strip()
        answer_sentence = data.get("answer_sentence", "").strip()
        is_correct = data.get("is_correct", False)
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() in ("true", "1", "yes")
        opposite_answer = data.get("opposite_answer", "").strip()
        return {
            "final_answer": final_answer or item.get("output", "")[:200],
            "answer_sentence": answer_sentence or final_answer,
            "is_correct": is_correct,
            "opposite_answer": opposite_answer or (item.get("answer", "") if not is_correct else final_answer),
        }
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Failed to parse API response: {e}. Content: {content[:300]}")
        return {
            "final_answer": "",
            "answer_sentence": "",
            "is_correct": False,
            "opposite_answer": item.get("answer", ""),
        }


def _fill_mask_template(template: str, sentence_a: str, sentence_b: str) -> str:
    """Insert report/sentence text without str.format (avoids KeyError when template uses {answer}/{output} or contains JSON examples)."""
    return (
        template.replace("{sentence_a}", sentence_a)
        .replace("{sentence_b}", sentence_b)
        .replace("{answer}", sentence_a)
        .replace("{output}", sentence_b)
    )


def _parse_mask_response(content: str, sentence_a: str, sentence_b: str) -> tuple[str, str]:
    """Parse GPT response for masked sentences; return (masked_a, masked_b)."""
    try:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        masked_a = (data.get("sentence_a_masked") or data.get("masked_sentence_1") or "").strip()
        masked_b = (data.get("sentence_b_masked") or data.get("masked_sentence_2") or "").strip()
        if not masked_a:
            masked_a = sentence_a
        if not masked_b:
            masked_b = sentence_b
        return (masked_a, masked_b)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Failed to parse mask response: {e}. Content: {content[:300]}")
        return (sentence_a, sentence_b)


async def mask_differing_phrases(
    sentence_a: str,
    sentence_b: str,
    mask_template: str,
    *,
    max_tokens: int = 300,
    model: str | None = None,
    debug_log: list[dict] | None = None,
    debug_batch: int = 0,
    debug_sample_id: object = None,
) -> Optional[tuple[str, str]]:
    """Call GPT to wrap differing phrases in <mask></mask> in both sentences."""
    eval_model = model if model is not None else _evaluation_model()
    text = _fill_mask_template(mask_template, sentence_a, sentence_b)
    message = {"role": "user", "content": [{"type": "text", "text": text}]}
    response = await _chat_completion_create(
        model=eval_model,
        messages=[message],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    if response is None:
        return None
    content = response.choices[0].message.content or ""
    if debug_log is not None:
        _debug_log_append(
            debug_log,
            batch=debug_batch,
            sample_id=debug_sample_id,
            step=20,
            label="MASK prompt (full)",
            content=text,
        )
        _debug_log_append(
            debug_log,
            batch=debug_batch,
            sample_id=debug_sample_id,
            step=21,
            label="MASK response (full)",
            content=content,
        )
    return _parse_mask_response(content, sentence_a, sentence_b)


async def main(
    csv_path: str,
    train_data: list[dict],
    prompt_config: RRPOPromptConfig | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    if prompt_config is None:
        prompt_config = DEFAULT_RRPO_PROMPT_CONFIG
    preprocess_rrpo_dataset_answers(train_data)
    preference_data: list[dict] = []
    debug_log: list[dict] | None = [] if is_debug else None
    for i in range(0, len(train_data), batch_size):
        print(f"Processing batch {i // batch_size + 1} / {(len(train_data) + batch_size - 1) // batch_size}")

        batch = train_data[i : i + batch_size]
        batch_num = i // batch_size

        async def locate(item) -> Optional[dict]:
            question = item["question"]
            output = item["output"]
            answer = item["answer"]
            text = prompt_config.locate_template.format(
                question=question, answer=answer, output=output
            )
            message = {"role": "user", "content": [{"type": "text", "text": text}]}
            response = await _chat_completion_create(
                model=_evaluation_model(),
                messages=[message],
                temperature=0.0,
                max_tokens=400,
            )
            if response is None:
                return None
            content = response.choices[0].message.content or ""
            if debug_log is not None:
                sid = item.get("id")
                _debug_log_append(
                    debug_log,
                    batch=batch_num,
                    sample_id=sid,
                    step=2,
                    label="VQA LOCATE prompt (full)",
                    content=text,
                )
                _debug_log_append(
                    debug_log,
                    batch=batch_num,
                    sample_id=sid,
                    step=3,
                    label="VQA LOCATE response (full)",
                    content=content,
                )
            parsed = _parse_single_api_response(content, item)
            print(
                f"Question: {question}, \n"
                f"Given Answer: {output}, \n"
                f"Final: {parsed['final_answer']}, \n"
                f"Sentence: {parsed['answer_sentence']}, \n"
                f"Correct: {parsed['is_correct']}, \n"
                f"Opposite: {parsed['opposite_answer']}"
            )
            print("-" * 100)
            return parsed

        tasks = [locate(item) for item in batch]
        results = await asyncio.gather(*tasks)
        for j, res in enumerate(results):
            if res is None:
                continue
            train_data[i + j]["final_answer"] = res["final_answer"]
            train_data[i + j]["is_correct"] = res["is_correct"]
            train_data[i + j]["answer_sentence"] = res["answer_sentence"]
            train_data[i + j]["opposite_answer"] = res["opposite_answer"]

        async def mask_one(res: Optional[dict], item: dict) -> Optional[dict]:
            if res is None:
                return None
            sent_a = res["answer_sentence"]
            sent_b = res["opposite_answer"]
            if not sent_a or not sent_b:
                return None
            mask_out = await mask_differing_phrases(
                sent_a,
                sent_b,
                prompt_config.mask_template,
                debug_log=debug_log,
                debug_batch=batch_num,
                debug_sample_id=item.get("id"),
            )
            if mask_out is None:
                return None
            masked_a, masked_b = mask_out
            was_correct = res["is_correct"]
            if was_correct:
                chosen, rejected = masked_a, masked_b
            else:
                chosen, rejected = masked_b, masked_a
            return {
                "id": item.get("id"),
                "qid": item.get("qid"),
                "prompt": item.get("question"),
                "chosen": chosen,
                "rejected": rejected,
                "was_correct": was_correct,
                "image_path": item.get("image_path"),
            }

        mask_tasks = [mask_one(results[j], train_data[i + j]) for j in range(len(results))]
        mask_results = await asyncio.gather(*mask_tasks)
        for rec in mask_results:
            if rec is not None:
                preference_data.append(rec)

        if is_debug:
            break

    if debug_log is not None:
        _print_debug_log(debug_log)

    return pd.DataFrame(train_data), preference_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build on-policy VQA preference pairs (prefer: python -m fire_mpo.pipeline.build_text_prefs)")
    parser.add_argument("--model", type=str, default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset", type=str, default="slake")
    parser.add_argument("--csv", type=str, required=True, help="On-policy inference CSV")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    cli = parser.parse_args()

    dataset_name = cli.dataset
    model_name = cli.model
    csv_path = cli.csv
    json_path = cli.output or str(
        Path(__file__).resolve().parents[2]
        / "preference_dataset"
        / model_name
        / dataset_name
        / "greedy"
        / "rrpo.json"
    )
    if os.path.exists(json_path) and not cli.force:
        raise FileExistsError(f"Preference JSON already exists: {json_path}")
    prompt_config = get_rrpo_prompt_config()

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    train_data = df.to_dict("records")

    result_df, preference_data = asyncio.run(main(csv_path, train_data, prompt_config))

    # Postprocess chosen/rejected before saving
    for rec in preference_data:
        rec["chosen"] = _normalize_chosen_rejected(rec["chosen"])
        rec["rejected"] = _normalize_chosen_rejected(rec["rejected"])

    # Drop samples with no <mask> in either field
    def _has_mask_tags(text: str) -> bool:
        return "<mask>" in text and "</mask>" in text

    n_before = len(preference_data)
    preference_data = [
        rec
        for rec in preference_data
        if _has_mask_tags(rec["chosen"]) or _has_mask_tags(rec["rejected"])
    ]
    n_dropped = n_before - len(preference_data)
    if n_dropped:
        print(f"Dropped {n_dropped} samples with no <mask></mask> in chosen or rejected.")

    # Keep only one entry per (id, chosen, rejected)
    seen: set[tuple] = set()
    unique_data: list[dict] = []
    for rec in preference_data:
        key = (rec["id"], rec["chosen"], rec["rejected"])
        if key not in seen:
            seen.add(key)
            unique_data.append(rec)
    n_dup = len(preference_data) - len(unique_data)
    preference_data = unique_data
    if n_dup:
        print(f"Deduplicated by (id, chosen, rejected): kept {len(preference_data)} unique, removed {n_dup} duplicates.")

    output_path = csv_path.rsplit(".", 1)[0] + "_with_opp.csv"
    result_df.to_csv(output_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(preference_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(preference_data)} → {json_path}")
    print(f"Saved side CSV to {output_path}")
