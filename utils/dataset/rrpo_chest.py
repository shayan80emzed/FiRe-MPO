"""Chest X-ray report RRPO: Gemini rewrite then mask (two calls) for chosen vs rejected pairs."""

import asyncio
import json
import os
import re
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

batch_size = 25
is_debug = False

# RRPO chest rewrite: which CSV field is Report A (rewritten base) vs Report B (context for edits).
# False: answer = Report A, output = Report B (default).
# True: output = Report A, answer = Report B.
BASE_REPORT_FROM_OUTPUT = True


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3-flash-preview"
GEN_MAX_ATTEMPTS = 3
GEN_RETRY_DELAY_SEC = 30

CHEST_REWRITE_PROMPT = """**Task:** Rewrite **Report A** to make a medically different and difficult rejected pair.

**Instruction:** Replace a few medical words in Report A to create a fine-grained rejected pair. Use the facts in **Report B** when possible.

**Constraints:**
* **Structural Locking:** The format of Report A should be preserved.
* **Token Parsimony:** At most **two or three words per sentence** can be changed.
* **Output Format:** Return **only** the rewritten report.

**Input Data:**
* **Report A:** {report_a}
* **Report B:** {report_b}
"""

# Used when Report A is the model output field (``output``): JSON with ``all_correct`` + ``rewritten_report_a``.
CHEST_REWRITE_PROMPT_OUTPUT_BASE = """**Task:** Rewrite **Report A** to make a medically difficult preference pair.

**Instruction:**
* **Change Logic**: Use the facts in **Report B** to replace a few medical words in Report A to create a fine-grained preferred pair.
* **all_correct**: If all the fact in Report B are already present in Report A, change a few words yourself to make a rejected pair.

**Constraints:**
* **Structural Locking:** The format of Report A should be preserved.
* **Token Parsimony:** At most **two or three words per sentence** can be changed.
* **Clinical Significance:** The rejected response should be clinically different with Report A. Changing synonyms (e.g. "clear" vs "normal", "healthy" vs "unremarkable") is not accepted.
* **Output Format:** Return a JSON object containing keys `all_correct` and `rewritten_report_a`.

**Input Data:**
* **Report A:** {report_a}
* **Report B:** {report_b}
"""

CHEST_MASK_PROMPT = """Compare two reports and put the differing phrases inside `<mask>` `</mask>` tags.

Output:
Return **only** a valid JSON object with two keys: `report_a_masked` and `report_b_masked`. Each key should be a string with the differing phrases wrapped in `<mask>` `</mask>` tags.

Report A:
{report_a}

Report B:
{rewritten_report_a}
"""


class ChestRewriteJsonOutput(BaseModel):
    all_correct: bool = Field(
        description="True if all facts from Report B were already present in Report A."
    )
    rewritten_report_a: str = Field(
        description="Rewritten Report A for the preference pair (rejected side before masking)."
    )


class ChestMaskedPairOutput(BaseModel):
    report_a_masked: str = Field(
        description="Report A with <mask></mask> on differing phrases vs the other report."
    )
    report_b_masked: str = Field(
        description="The second report with <mask></mask> on differing phrases."
    )


def preprocess_rrpo_dataset_answers(train_data: list[dict]) -> None:
    """Remove placeholder token XXXX from ground-truth answers (mutates rows in place)."""
    for row in train_data:
        a = row.get("answer")
        if not isinstance(a, str) or not a:
            continue
        row["answer"] = a.replace("XXXX", "")


def _normalize_mask_tag_variants(s: str) -> str:
    """Map nonstandard mask-like tags (e.g. <maskdetails>) to <mask> and </mask>."""
    if not s:
        return s
    s = re.sub(
        r"(?i)<\s*mask[\w-]*(?:\s[^>]*)?>",
        "<mask>",
        s,
    )
    s = re.sub(
        r"(?i)</\s*mask[\w-]*\s*>",
        "</mask>",
        s,
    )
    return s


def _normalize_chosen_rejected(s: str) -> str:
    """Postprocess chosen/rejected: capital, <mask>/</mask> spacing, period end."""
    if not s or not s.strip():
        return s
    s = s.strip()
    s = _normalize_mask_tag_variants(s)
    s = re.sub(r"\s+<mask>", " <mask>", s)
    s = re.sub(r"<mask>\s+", "<mask>", s)
    s = re.sub(r"\s+</mask>", "</mask>", s)
    s = re.sub(r"</mask>\s+", "</mask> ", s)
    s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
    s = s.rstrip()
    if s.endswith(","):
        s = s[:-1] + "."
    elif s and s[-1] not in ".!?":
        s = s + "."
    return s


def _mask_spans_consistent(chosen: str, rejected: str) -> bool:
    """True iff each side has balanced <mask></mask> counts and both sides have the same number of spans."""
    o_c, c_c = chosen.count("<mask>"), chosen.count("</mask>")
    o_r, c_r = rejected.count("<mask>"), rejected.count("</mask>")
    if o_c != c_c or o_r != c_r:
        return False
    # if o_c != o_r:
    #     return False
    return o_c > 0


def _gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise ValueError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini (chest RRPO pipeline)."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def _fill_chest_rewrite_prompt(
    *, report_a: str, report_b: str, output_as_base: bool
) -> str:
    template = (
        CHEST_REWRITE_PROMPT_OUTPUT_BASE if output_as_base else CHEST_REWRITE_PROMPT
    )
    return template.replace("{report_a}", report_a).replace("{report_b}", report_b)


def _fill_chest_mask_prompt(*, report_a: str, rewritten_report_a: str) -> str:
    return (
        CHEST_MASK_PROMPT.replace("{report_a}", report_a).replace(
            "{rewritten_report_a}", rewritten_report_a
        )
    )


async def _generate_chest_rewrite(
    client: genai.Client,
    *,
    report_a: str,
    report_b: str,
    output_as_base_report: bool,
    debug_sample_id: object,
) -> Optional[tuple[str, bool]]:
    """Returns (rewritten_report_a, all_correct). ``all_correct`` is always True when not using JSON rewrite."""
    contents = _fill_chest_rewrite_prompt(
        report_a=report_a, report_b=report_b, output_as_base=output_as_base_report
    )
    if output_as_base_report:
        config = types.GenerateContentConfig(
            temperature=1,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            response_mime_type="application/json",
            response_json_schema=ChestRewriteJsonOutput.model_json_schema(),
        )
    else:
        config = types.GenerateContentConfig(
            temperature=1,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        )
    last_error: Optional[BaseException] = None
    for attempt in range(GEN_MAX_ATTEMPTS):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                print(f"Gemini rewrite: empty response; id={debug_sample_id!r}")
                return None
            if output_as_base_report:
                parsed = ChestRewriteJsonOutput.model_validate_json(text)
                rewritten = (parsed.rewritten_report_a or "").strip()
                if not rewritten:
                    print(
                        f"Gemini rewrite: empty rewritten_report_a; id={debug_sample_id!r}"
                    )
                    return None
                return (rewritten, parsed.all_correct)
            return (text, True)
        except Exception as e:
            last_error = e
            print(
                f"Gemini rewrite failed (attempt {attempt + 1}/{GEN_MAX_ATTEMPTS}): {e}"
            )
            if attempt < GEN_MAX_ATTEMPTS - 1:
                await asyncio.sleep(GEN_RETRY_DELAY_SEC)
    print(f"Skipping Gemini rewrite after {GEN_MAX_ATTEMPTS} failures: {last_error}")
    return None


async def _generate_chest_mask(
    client: genai.Client,
    *,
    report_a: str,
    rewritten_report_a: str,
    debug_sample_id: object,
) -> Optional[ChestMaskedPairOutput]:
    contents = _fill_chest_mask_prompt(
        report_a=report_a, rewritten_report_a=rewritten_report_a
    )
    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        response_mime_type="application/json",
        response_json_schema=ChestMaskedPairOutput.model_json_schema(),
    )
    last_error: Optional[BaseException] = None
    for attempt in range(GEN_MAX_ATTEMPTS):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                print(f"Gemini mask: empty response; id={debug_sample_id!r}")
                return None
            return ChestMaskedPairOutput.model_validate_json(text)
        except Exception as e:
            last_error = e
            print(f"Gemini mask failed (attempt {attempt + 1}/{GEN_MAX_ATTEMPTS}): {e}")
            if attempt < GEN_MAX_ATTEMPTS - 1:
                await asyncio.sleep(GEN_RETRY_DELAY_SEC)
    print(f"Skipping Gemini mask after {GEN_MAX_ATTEMPTS} failures: {last_error}")
    return None


async def _generate_chest_masked(
    client: genai.Client,
    *,
    report_a: str,
    report_b: str,
    output_as_base_report: bool,
    debug_log: list[dict] | None,
    debug_batch: int,
    debug_sample_id: object,
) -> Optional[tuple[ChestMaskedPairOutput, bool]]:
    rewrite_out = await _generate_chest_rewrite(
        client,
        report_a=report_a,
        report_b=report_b,
        output_as_base_report=output_as_base_report,
        debug_sample_id=debug_sample_id,
    )
    if rewrite_out is None:
        return None
    rewritten, all_correct = rewrite_out
    masked = await _generate_chest_mask(
        client,
        report_a=report_a,
        rewritten_report_a=rewritten,
        debug_sample_id=debug_sample_id,
    )
    if masked is None:
        return None
    return (masked, all_correct)


async def main(
    csv_path: str,
    train_data: list[dict],
    *,
    base_report_from_output: bool | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build RRPO pairs: rewrite then mask via Gemini.

    Report A is the text that gets rewritten; Report B supplies facts for fine-grained edits.
    Mapping is controlled by ``BASE_REPORT_FROM_OUTPUT`` (module default) or
    ``base_report_from_output`` (per-call override when not None).
    """
    use_output_as_base = (
        BASE_REPORT_FROM_OUTPUT
        if base_report_from_output is None
        else base_report_from_output
    )
    preprocess_rrpo_dataset_answers(train_data)
    preference_data: list[dict] = []
    ok_count = 0
    debug_log: list[dict] | None = [] if is_debug else None
    client = _gemini_client()

    for i in range(0, len(train_data), batch_size):
        print(f"Processing batch {i // batch_size + 1} / {(len(train_data) + batch_size - 1) // batch_size}")

        batch = train_data[i : i + batch_size]
        batch_num = i // batch_size

        async def one(item: dict) -> Optional[dict]:
            answer_s = (item.get("answer") or "").strip()
            output_s = (item.get("output") or "").strip()
            if not answer_s or not output_s:
                print(f"Chest RRPO: missing answer or output; skipping id={item.get('id')!r}.")
                return None
            if use_output_as_base:
                report_a, report_b = output_s, answer_s
            else:
                report_a, report_b = answer_s, output_s
            out = await _generate_chest_masked(
                client,
                report_a=report_a,
                report_b=report_b,
                output_as_base_report=use_output_as_base,
                debug_log=debug_log,
                debug_batch=batch_num,
                debug_sample_id=item.get("id"),
            )
            if out is None:
                print(f"Chest RRPO: Gemini failed; skipping id={item.get('id')!r}.")
                return None
            masked_pair, was_correct = out
            tagged_a = _normalize_mask_tag_variants(masked_pair.report_a_masked.strip())
            tagged_rw = _normalize_mask_tag_variants(masked_pair.report_b_masked.strip())
            combined = (tagged_a + tagged_rw).lower()
            if "<mask>" not in combined or "</mask>" not in combined:
                print(tagged_a)
                print(tagged_rw)
                print(f"Chest RRPO: no <mask></mask> in structured output; skipping id={item.get('id')!r}.")
                return None
            if not _mask_spans_consistent(tagged_a, tagged_rw):
                print(
                    f"Chest RRPO: unequal or unbalanced <mask></mask> "
                    f"(chosen open={tagged_a.count('<mask>')} close={tagged_a.count('</mask>')}, "
                    f"rejected open={tagged_rw.count('<mask>')} close={tagged_rw.count('</mask>')}); "
                    f"skipping id={item.get('id')!r}."
                )
                return None
            # When Report A is ``output``: if all_correct, prefer original output; else prefer rewrite.
            if use_output_as_base:
                if was_correct:
                    chosen, rejected = tagged_a, tagged_rw
                else:
                    chosen, rejected = tagged_rw, tagged_a
            else:
                chosen, rejected = tagged_a, tagged_rw
            return {
                "id": item.get("id"),
                "qid": item.get("qid"),
                "prompt": item.get("question"),
                "chosen": chosen,
                "rejected": rejected,
                "was_correct": was_correct,
                "image_path": item.get("image_path"),
            }

        results = await asyncio.gather(*[one(item) for item in batch])
        for rec in results:
            if rec is not None:
                preference_data.append(rec)
                ok_count += 1

        if is_debug:
            break

    print(f"Chest RRPO summary: {ok_count} preference rows produced (model={MODEL_NAME}).")

    return pd.DataFrame(train_data), preference_data


if __name__ == "__main__":
    dataset_name = "iu_xray"
    model_name = "Qwen3-VL-4B-Instruct"
    csv_path = f"/home/emzed/projects/aip-dolatab6/emzed/med-align/experiments/{dataset_name}_{model_name}/None_1_train.csv"
    json_path = f"/home/emzed/projects/aip-dolatab6/emzed/med-align/preference_dataset/{model_name}/{dataset_name}/greedy/rrpo.json"
    if os.path.exists(json_path):
        raise FileExistsError(f"Preference JSON already exists: {json_path}")

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    train_data = df.to_dict("records")

    # Toggle BASE_REPORT_FROM_OUTPUT at module top, or pass base_report_from_output=...
    result_df, preference_data = asyncio.run(main(csv_path, train_data))

    for rec in preference_data:
        rec["chosen"] = _normalize_chosen_rejected(rec["chosen"])
        rec["rejected"] = _normalize_chosen_rejected(rec["rejected"])

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

    n_before = len(preference_data)
    preference_data = [
        rec for rec in preference_data if _mask_spans_consistent(rec["chosen"], rec["rejected"])
    ]
    n_dropped_masks = n_before - len(preference_data)
    if n_dropped_masks:
        print(
            f"Dropped {n_dropped_masks} samples with unequal or unbalanced <mask></mask> "
            "between chosen and rejected."
        )

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
    print(f"Saved to {output_path}")

    with open(json_path, "w") as f:
        json.dump(preference_data, f, indent=2)
    print(f"Saved preference JSON ({len(preference_data)} entries) to {json_path}")
