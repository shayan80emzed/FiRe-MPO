"""
DPO Dataset for Medical VQA Training
Based on web implementation for proper DPO training with vision-language models
"""

import json
import os
import logging
from typing import Dict, Any, List
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from templates.prompt_templates import prompt_templates
from templates.conversation_templates import conversation_templates

logger = logging.getLogger(__name__)


class DPODataset(Dataset):
    """Custom DPO dataset for medical VQA training"""
    
    def __init__(self, data_path: str, image_folder: str = None, max_samples: int = None):
        """
        Initialize the DPO dataset
        
        Args:
            data_path: Path to the JSON file containing DPO data
            image_folder: Base directory containing images
            max_samples: Maximum number of samples to load
        """
        self.image_folder = image_folder
        
        # Load raw data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Convert to DPO format for HuggingFace Dataset using conversation templates
        dpo_data = []
        for item in raw_data:
            if max_samples is not None and len(dpo_data) >= max_samples:
                break
            
            # Construct full image path (chosen / winner image m_w)
            if image_folder is not None:
                full_image_path = os.path.join(image_folder, item["image_path"])
            else:
                full_image_path = item["image_path"]

            rej_img = item.get("rejected_image_path") or item.get("image_path_rejected")
            full_rejected_image_path = None
            if rej_img:
                if image_folder is not None and not os.path.isabs(rej_img):
                    full_rejected_image_path = os.path.join(image_folder, rej_img)
                else:
                    full_rejected_image_path = rej_img

            # Use conversation templates to create DPO conversation
            dpo_conversation = conversation_templates.get_qwen().create_dpo_conversation(
                image_path=full_image_path,
                prompt=item["prompt"],
                chosen=item["chosen"],
                rejected=item["rejected"],
                rejected_image_path=full_rejected_image_path,
            )
            
            dpo_data.append(dpo_conversation)
        
        # Initialize HuggingFace Dataset
        temp_dataset = Dataset.from_list(dpo_data)
        super().__init__(
            arrow_table=temp_dataset._data,
            info=temp_dataset.info,
            split=temp_dataset.split
        )
        
        logger.info(f"Loaded {len(self)} DPO examples from {data_path}")