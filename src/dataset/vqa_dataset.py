import csv
import json
import os
from pathlib import Path
from PIL import Image

DATASET_DIR = "/data/shayan/datasets"
REASONING_DIR = 'Medthink_Dataset'
IMAGE_FOLDER_PREFIX = {
    "vqa_rad": None,
    "slake": "images",
    # "slake": None,
    "path_vqa": None,
    "iu_xray": "iu_xray/images",
    "pmc_vqa": "images",
    "vgmed": None,
}

DEFAULT_KEYS = {
    "id": "id",
    "image_path": "image_path",
    "question": "question",
    "answer": "answer",
}

JSON_FILES_FORMAT = {
    "vqa_rad": lambda split: f"vqa_rad_{split}.json",
    "slake": lambda split: f"{split}.json",
    "path_vqa": lambda split: f"{split}.json",
    "iu_xray": lambda split: f"{split}.json",
}

# Datasets loaded from CSV instead of JSON (see ``VQADataset.__init__``).
PMC_VQA_TEST_CSV = "test_clean.csv"

# VGMED: test-only JSON under ``DATASET_DIR``; filenames fixed (see ``VQADataset.__init__``).
VGMED_DIR = "VGMED"
VGMED_LOC_JSON = "VGMED_loc_slake.json"
VGMED_ATT_JSON = "VGMED_att_slake.json"

# MedXpertQA MM (TsinghuaC3I/MedXpertQA): HF ``MM/dev.jsonl`` is the development / train split;
# ``MM/test.jsonl`` is the test split. Images live under ``images/`` after ``images.zip`` is extracted.
MEDXPERTQA_DIR = "MedXpertQA"
MEDXPERTQA_MM_JSONL = {"train": "dev.jsonl", "test": "test.jsonl"}


def _supported_dataset_names():
    return set(JSON_FILES_FORMAT) | {"pmc_vqa", "vgmed", "medxpertqa_mm"}

# IU-Xray JSON may list several views in ``image_paths``; labeling scripts set this relative
# path (under iu_xray/images) for the chosen frontal view. If absent, ``image_path`` is used.
IU_XRAY_PRIMARY_IMAGE_KEY = "primary_image_relpath"


class VQADataset:
    def __init__(self, dataset_name, split, reasoning=False, subset=None):
        assert dataset_name in _supported_dataset_names(), f"Dataset {dataset_name} not supported"
        if subset is not None and dataset_name != "vgmed":
            raise ValueError("subset is only supported when dataset_name='vgmed'")
        if dataset_name == "vgmed" and subset is not None and subset not in ("loc", "att"):
            raise ValueError('subset must be None, "loc", or "att"')
        self.dataset_name = dataset_name
        self.reasoning = reasoning
        self.subset = subset
        if reasoning:
            dataset_dir = Path(DATASET_DIR) / REASONING_DIR
        else:
            dataset_dir = Path(DATASET_DIR)

        if dataset_name == "pmc_vqa":
            if reasoning:
                raise ValueError("pmc_vqa does not support reasoning=True")
            if split != "test":
                raise ValueError('pmc_vqa only supports split="test" (uses test_clean.csv)')
            csv_path = dataset_dir / "pmc_vqa" / PMC_VQA_TEST_CSV
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.data = []
                for idx, row in enumerate(reader):
                    self.data.append(
                        {
                            "id": str(idx),
                            "qid": idx,
                            "image_path": row["Figure_path"].strip(),
                            "question": row["Question"].strip(),
                            "answer": row["Answer"].strip(),
                        }
                    )
        elif dataset_name == "vgmed":
            if reasoning:
                raise ValueError("vgmed does not support reasoning=True")
            if split != "test":
                raise ValueError('vgmed only supports split="test"')
            vgmed_dir = dataset_dir / VGMED_DIR
            if subset == "loc":
                json_paths = [vgmed_dir / VGMED_LOC_JSON]
            elif subset == "att":
                json_paths = [vgmed_dir / VGMED_ATT_JSON]
            else:
                json_paths = [vgmed_dir / VGMED_LOC_JSON, vgmed_dir / VGMED_ATT_JSON]
            self.data = []
            for json_path in json_paths:
                with open(json_path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
                offset = len(self.data)
                for i, item in enumerate(chunk):
                    self.data.append(
                        {
                            "id": str(offset + i),
                            "qid": offset + i,
                            "image_path": item["image"],
                            "question": item["question"],
                            # "answer": item["label"],
                            "answer": "Yes",
                            "bbox": item.get("bbox") or [],
                        }
                    )
        elif dataset_name == "medxpertqa_mm":
            if reasoning:
                raise ValueError("medxpertqa_mm does not support reasoning=True")
            if split not in MEDXPERTQA_MM_JSONL:
                raise ValueError(
                    'medxpertqa_mm only supports split="train" (HF dev) or split="test"'
                )
            jsonl_path = dataset_dir / MEDXPERTQA_DIR / "MM" / MEDXPERTQA_MM_JSONL[split]
            self.data = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for qid, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    images = rec.get("images") or []
                    if not images:
                        continue
                    label = (rec.get("label") or "").strip()
                    options = rec.get("options") or {}
                    answer = options.get(label, label)
                    self.data.append(
                        {
                            "id": rec["id"],
                            "qid": qid,
                            "image_path": images[0],
                            "question": rec["question"],
                            "answer": answer,
                        }
                    )
        elif not reasoning:
            json_path = dataset_dir / dataset_name / JSON_FILES_FORMAT[dataset_name](split)
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            json_path = dataset_dir / dataset_name / (split + ".json")
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

        if dataset_name == "vgmed":
            self.image_dir = dataset_dir / VGMED_DIR
        elif dataset_name == "medxpertqa_mm":
            self.image_dir = dataset_dir / MEDXPERTQA_DIR / "images"
        else:
            self.image_dir = dataset_dir / dataset_name
            if IMAGE_FOLDER_PREFIX[dataset_name] is not None:
                self.image_dir = self.image_dir / IMAGE_FOLDER_PREFIX[dataset_name]

        self.default_keys = DEFAULT_KEYS.copy()
        self.to_rgb = False

        if dataset_name == "vqa_rad":
            pass
        elif dataset_name == "slake":
            self.default_keys["id"] = "img_id"
            self.default_keys["image_path"] = "img_name"
            pass
        elif dataset_name == "path_vqa":
            self.to_rgb = True
        elif dataset_name == "iu_xray":
            self.default_keys["question"] = "hf_question"
        elif dataset_name == "pmc_vqa":
            pass
        elif dataset_name == "vgmed":
            pass
        elif dataset_name == "medxpertqa_mm":
            pass
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.dataset_name == "iu_xray" and item.get(IU_XRAY_PRIMARY_IMAGE_KEY):
            rel = Path(item[IU_XRAY_PRIMARY_IMAGE_KEY])
        else:
            rel = Path(item[self.default_keys["image_path"]])
        if rel.is_absolute():
            image_path = rel
        else:
            image_path = self.image_dir / rel
        output = {
            "id": item[self.default_keys["id"]],
            "qid": item.get("qid", idx),
            "image_path": str(image_path),
            "question": item[self.default_keys["question"]],
            "answer": item[self.default_keys["answer"]],
            # "image": Image.open(image_path).convert("RGB") if self.to_rgb else Image.open(image_path),
        }
        if self.reasoning:
            output['reasoning'] = item['reasoning']
        if self.dataset_name == "vgmed" and "bbox" in item:
            output["bbox"] = item["bbox"]
        return output
        
    def get_dataloader(self, batch_size, shuffle=False):
        indices = list(range(len(self)))
        if shuffle:
            import random
            random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start+batch_size]
            batch = [self[i] for i in batch_indices]
            yield batch

