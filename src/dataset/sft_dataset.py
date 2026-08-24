"""
DPO Dataset for Medical VQA Training
Based on web implementation for proper DPO training with vision-language models
"""

import json
import os
import logging
from datasets import Dataset
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from templates.conversation_templates import conversation_templates

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    
    """Custom SFT dataset on chosen responses of DPO dataset for medical VQA training"""
    
    def __init__(self, data_path: str, image_folder: str = None, max_samples: int = None):
        self.image_folder = image_folder
        
        # Load raw data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Convert to SFT format for HuggingFace Dataset using conversation templates
        sft_data = []
        for item in raw_data:
            if max_samples is not None and len(sft_data) >= max_samples:
                break
            
            # Construct full image path
            if image_folder is not None:
                full_image_path = os.path.join(image_folder, item["image_path"])
            else:
                full_image_path = item["image_path"]
            
            # Use conversation templates to create SFT conversation
            sft_conversation = conversation_templates.get_qwen().create_sft_conversation(
                image_path=full_image_path,
                prompt=item["prompt"],
                chosen=item["chosen"]
            )
            
            sft_data.append(sft_conversation)
        
        # Initialize HuggingFace Dataset
        temp_dataset = Dataset.from_list(sft_data)
        super().__init__(
            arrow_table=temp_dataset._data,
            info=temp_dataset.info,
            split=temp_dataset.split
        )
        
        logger.info(f"Loaded {len(self)} SFT examples from {data_path}")
        