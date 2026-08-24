"""
Conversation Templates for Medical Alignment Project
Model-specific conversation templates for causal multimodal models
Each model has its own class based on actual usage patterns in the codebase
"""

from typing import Dict, List, Any, Optional, Union
from PIL import Image
from .prompt_templates import prompt_templates


class Qwen25VLConversationTemplates:
    """Conversation templates for Qwen2.5-VL models - used in inference and DPO training"""
    
    def __init__(self):
        self.model_name = "Qwen2.5-VL"
        self.system_role = prompt_templates.qwen_system_role
    
    def create_inference_conversation(self, image: Union[str, Image.Image], question: str) -> List[Dict[str, Any]]:
        """
        Create Qwen2.5-VL inference conversation format (used in qwen25_inference.py)
        
        Args:
            image: Image path or PIL Image object
            question: Question about the image
            
        Returns:
            List of messages in Qwen2.5-VL conversation format
        """
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_role
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    
    def create_dpo_conversation(
        self,
        image_path: str,
        prompt: str,
        chosen: str,
        rejected: str,
        rejected_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create Qwen2.5-VL DPO training conversation (used in dpo_dataset.py)
        
        Args:
            image_path: Path to the chosen (winner) image m_w
            prompt: Question about the image
            chosen: Preferred response
            rejected: Non-preferred response
            rejected_image_path: Optional path to the rejected image m_l for visual-preference RRPO terms
            
        Returns:
            Dictionary with prompt, chosen, rejected conversations, and images list
        """
        # Format prompt using the template from dpo_dataset.py
        prompt_messages = [
            # {
            #     "role": "system",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": prompt_templates.qwen_system_message()
            #         }
            #     ]
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        # "text": None,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Format chosen response
        chosen_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": chosen
                    }
                ]
            }
        ]
        
        # Format rejected response
        rejected_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": rejected
                    }
                ]
            }
        ]
        
        out: Dict[str, Any] = {
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "images": [image_path],  # DPO expects list of image paths (chosen / m_w)
        }
        if rejected_image_path:
            out["rejected_image_path"] = rejected_image_path
        return out
        
    def create_sft_conversation(self, image_path: str, prompt: str, chosen: str) -> Dict[str, List[Dict[str, Any]]]:    
        output_messages = self.create_dpo_conversation(image_path, prompt, chosen, "")
        del output_messages["rejected"]
        output_messages["completion"] = output_messages["chosen"]
        del output_messages["chosen"]
        return output_messages
        
    
    def batch_inference_conversations(self, batch: List[Dict[str, Any]], 
                                    image_key: str = "image", text_key: str = "question") -> List[List[Dict[str, Any]]]:
        """Create batch of Qwen2.5-VL inference conversations"""
        return [
            self.create_inference_conversation(item[image_key], item[text_key])
            for item in batch
        ]


class GPT4oConversationTemplates:
    """Conversation templates for GPT-4o models - used in refinement tasks"""
    
    def __init__(self):
        self.model_name = "GPT-4o"
        self.system_role = prompt_templates.medical_refinement_system_role
    
    def create_refinement_conversation(self, image: Image.Image, question: str, correct_answer: str, model_output: str) -> List[Dict[str, Any]]:
        """
        Create GPT-4o refinement conversation (used in preference_construction.py)
        
        Args:
            image: PIL Image object
            question: Original question
            correct_answer: Ground truth answer
            model_output: Model's incorrect output
            
        Returns:
            List of messages in GPT-4o conversation format
        """
        # Encode image to base64 (from preference_construction.py)
        import base64
        from io import BytesIO
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        image_url = f"data:image/jpeg;base64,{img_str}"
        
        refinement_prompt = prompt_templates.medical_refinement_prompt(question, correct_answer, model_output)
        
        return [
            # {
            #     "role": "system",
            #     "content": prompt_templates.medical_refinement_system_role
            # },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": refinement_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
    
    def create_opposite_conversation(self, question: str, correct_answer: str, model_output: str, is_correct: bool) -> List[Dict[str, Any]]:
        prompt = prompt_templates.medical_text_only_refinement_prompt(question, correct_answer, model_output, is_correct)
        
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]


class GPT4oMiniConversationTemplates:
    """Conversation templates for GPT-4o-mini models - used in correctness evaluation"""
    
    def __init__(self):
        self.model_name = "GPT-4o-mini"
        self.system_role = prompt_templates.medical_evaluation_system_role
    
    def create_evaluation_conversation(self, question: str, correct_answer: str, model_output: str) -> List[Dict[str, Any]]:
        """
        Create GPT-4o-mini evaluation conversation (used in correctness_evaluator.py)
        
        Args:
            question: Original question
            correct_answer: Ground truth answer
            model_output: Model's generated output
            
        Returns:
            List of messages in GPT-4o-mini conversation format
        """
        evaluation_prompt = prompt_templates.medical_evaluation_prompt(question, correct_answer, model_output)
        
        return [
            # {
            #     "role": "system",
            #     "content": prompt_templates.medical_evaluation_system_role
            # },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": evaluation_prompt}
                ]
            }
        ]
        
    def create_highlighter_conversation(self, question: str, model_output: str) -> List[Dict[str, Any]]:
        highlighter_prompt = prompt_templates.medical_highlighter_prompt(question, model_output)
        return [
            {
                "role": "system",
                "content": prompt_templates.medical_highlighter_system_role
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": highlighter_prompt}]
            }
        ]

    def create_reasoning_section_break_conversation(self, question: str, reasoning: str) -> List[Dict[str, Any]]:
        """
        Create GPT-4o-mini reasoning section break conversation (used in reasoning_section_break.py)
        
        Args:
            question: Original question
            reasoning: Generated reasoning response
        """

        return [
            {
                "role": "system",
                "content": prompt_templates.medical_reasoning_section_break_system_role
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_templates.medical_reasoning_section_break_prompt(question, reasoning)}
                ]
            }
        ]

class ConversationTemplates:
    """Centralized conversation templates organized by model type"""
    
    def __init__(self):
        self.models = {
            "qwen2.5-vl": Qwen25VLConversationTemplates(),
            "qwen25vl": Qwen25VLConversationTemplates(),  # Alias
            "gpt-4o": GPT4oConversationTemplates(),
            "gpt4o": GPT4oConversationTemplates(),  # Alias
            "gpt-4o-mini": GPT4oMiniConversationTemplates(),
            "gpt4omini": GPT4oMiniConversationTemplates(),  # Alias
        }
    
    def get_qwen(self) -> Qwen25VLConversationTemplates:
        return self.models["qwen2.5-vl"]
    
    def get_gpt4o(self) -> GPT4oConversationTemplates:
        return self.models["gpt-4o"]
    
    def get_gpt4omini(self) -> GPT4oMiniConversationTemplates:
        return self.models["gpt-4o-mini"]
    
    def list_available_models(self) -> List[str]:
        return list(self.models.keys())


# Global instance
conversation_templates = ConversationTemplates()

