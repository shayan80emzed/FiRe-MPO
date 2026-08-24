from dataclasses import dataclass

@dataclass
class PromptTemplates:
    """Centralized prompt templates for the medical alignment project"""
    
    # qwen_system_role: str = "You are a helpful medical AI assistant. Please reason step by step and then answer with a short final answer in English."
#     qwen_system_role: str = """
# You are a careful medical vision-language assistant. Always respond in the required reasoning structure and follow the constraints below.

# TASK
# Given a medical image and a natural-language question (and optionally a short clinical text), generate a structured, clinically sound reasoning process and answer.

# REQUIRED OUTPUT FORMAT
# 1- Understanding the query
#    - Explain what the question is about and what type of information are needed for answering the question.
#    - This section can be multiple sentences; demonstrate your understanding of the problem.
#    - Put this section in <UNDERSTANDING>...</UNDERSTANDING> tags.

# 2- Perception
#    - Describe the key visual findings or absence of findings in the image that are relevant to answering the question.
#    - Focus on what is visible in relation to the question. You do not need to provide exhaustive details (side, size, etc.) unless relevant.
#    - All image related facts MUST be in this section.
#    - Put this section in <PERCEPTION>...</PERCEPTION> tags.

# 3- Interpretation
#    - Link the perceived findings to their clinical meaning or implication.
#    - Do not add any other perceptions, only interpret the perceived visual information from previous sections.
#    - Include uncertainty where appropriate (e.g., “suggestive of…”, “cannot exclude…”).
#    - Put this section in <INTERPRETATION>...</INTERPRETATION> tags.

# 4- Final answer
#    - Provide one concise and short answer to the question.
#    - Put this section in <ANSWER>...</ANSWER> tags.

# CONSTRAINTS
# - Base reasoning strictly on the image and provided text.
# - Use standardized, professional medical terminology.
# - Make sure to have the tags for each section.

# STYLE
# - Structured, clear, factual, and concise.
# - No emojis or extra sections.

# """
#     qwen_system_role: str = """
# You are a careful medical vision-language assistant. Always respond in the required reasoning structure and follow the constraints below.

# TASK: Given a medical image and a natural-language question (and optionally a short clinical text), generate a structured, clinically sound reasoning process and answer.

# REQUIRED OUTPUT FORMAT
# <UNDERSTANDING>
#    - Explain what the question is about and what type of information are needed for answering the question.
#    - This section can be multiple sentences; demonstrate your understanding of the problem.
# </UNDERSTANDING>

# <PERCEPTION>
#    - Describe the key visual findings or absence of findings in the image that are relevant to answering the question.
#    - Focus on what is visible in relation to the question. You do not need to provide exhaustive details (side, size, etc.) unless relevant.
#    - All image related facts MUST be in this section.
# </PERCEPTION>

# <INTERPRETATION>
#    - Link the perceived findings to their clinical meaning or implication.
#    - Do not add any other perceptions, only interpret the perceived visual information from previous sections.
#    - Include uncertainty where appropriate (e.g., “suggestive of…”, “cannot exclude…”).
# </INTERPRETATION>

# <ANSWER>
#    - Provide one concise and short answer to the question.
# </ANSWER>

# CONSTRAINTS
# - Base reasoning strictly on the image and provided text.
# - Use standardized, professional medical terminology.
# - Make sure to have the tags for each section.

# STYLE
# - Structured, clear, factual, and concise.
# - No emojis or extra sections.

# """

    qwen_system_role: str = """
You are a helpful medical AI assistant. Please answer the question based on the image and then give an explanation for your answer in English.

REQUIRED OUTPUT FORMAT
<ANSWER>
    - Provide one concise and short answer to the question.
</ANSWER>

<EXPLANATION>
    - Provide an explanation for your answer.
</EXPLANATION>
"""

    
    medical_evaluation_system_role: str = "You are a medical expert evaluating the correctness of a model's response to a medical question given the correct answer."
    medical_refinement_system_role: str = "You are a medical expert tasked with identifying and refining the incorrect parts of a model's response to a medical question about the given medical image."
    medical_reasoning_section_break_system_role: str = "You are a medical expert tasked with breaking down a response to a medical question into specified sections."

    def medical_evaluation_prompt(self, question: str, answer: str, output: str) -> str:
        return f"""Question: \"{question}\",
Ground Truth Answer: \"{answer}\",
Model Answer: \"{output}\",

Please carefully examine the triplet and determine if the model's answer is correct. Consider:
1. Is the model's answer to the question consistent with the ground truth answer?
2. Please only answer with True or False, nothing else.
"""

    def medical_refinement_prompt(self, question: str, answer: str, output: str) -> str:
#         return f"""Here is a medical question, the correct answer, and a model's given answer to the question:
# Question: "{question}"
# Correct Answer: "{answer}"
# Given Answer by the model: "{output}"

# Please carefully examine the triplet and do the following:
# 1. Identify the model's final answer from the given answer and put it in the <mask> and </mask> tags.
# 2. Determine if the final answer is correct based on the correct answer.
# 3. If the final answer is incorrect, put the correct final answer from the correct answer in the <mask> and </mask> tags in the "refined_output". If the final answer is correct, repeat the same final answer in the "refined_output".

# Respond with a JSON format containing:
# - "is_correct": boolean (True if the final answer is correct, False otherwise)
# - "refined_output": string (the final answer with the correct answer in <mask> and </mask> tags if the original final answer was incorrect)

# Example responses:
# {{
#   "is_correct": False,
#   "refined_output": "A B C <mask> L </mask> E F"
# }}
# {{
#   "is_correct": True,
#   "refined_output": "A B C <mask> L </mask>"
# }}
# """
        return f"""Here is a medical question, the correct answer, and a model's given answer to the question:
Question: "{question}"
Correct Answer: "{answer}"
Given Answer by the model: "{output}"




Please carefully examine the triplet and do the following:
1. Identify the incorrect parts of the Given Answer. Incorrect phrases consist of non-factual medical information, hallucinated information based on the image, incorrect steps, or wrong answer to the question based on the correct answer. Change as little as possible (i.e. if one phrace is incorrect in a sentence, do not change the entire sentence, only the incorrect phrase).
2. Output the Given Answer, but enclose each incorrect phrase in <mask> and </mask> tags. Every other phrase outside the <mask> and </mask> tags should be kept and included as is. This is the "predicted_answer".
3. For each <mask> ... </mask> region, replace the incorrect phrase with the correct information based on the correct answer and the medical image. Do not remove the <mask> and </mask> tags. Do not change any part of the output except phrases inside the <mask> and </mask> tags. This is the "refined_output".

Respond with a JSON format containing:
- "predicted_answer": string (the Given Answer with incorrect parts masked)
- "refined_output": string (the string that replaces the incorrect parts of the predicted_answer with the correct parts within the <mask> and </mask> tags)

Example responses:
{{
  "predicted_answer": "A B C <mask> D </mask> E F",
  "refined_output": "A B C <mask> L </mask> E F"
}}
{{
  "predicted_answer": "A B C <mask> D </mask> E <mask> F </mask>",
  "refined_output": "A B C <mask> L </mask> E <mask> P </mask>"
}}
{{
  "predicted_answer": "<mask> A </mask> B C <mask> D </mask> E F",
  "refined_output": "<mask> Y </mask> B C <mask> L </mask> E F"
}}
"""

    
    def medical_reasoning_section_break_prompt(self, question: str, reasoning: str) -> str:
      return f"""You are given a medical question and a reasoning response to the question. Please break down the following reasoning into specified sections and return the sectioned format. Make sure to have the tags for each section. Do not change or add anything to the content of the reasoning, only break it down into the specified sections.
      Here is the question:
      {question}
      Here is the reasoning:
      {reasoning}
      
      Here is the sectioned format:
      <UNDERSTANDING>
        - This section explains what the question is about and what type of information are needed for answering the question.
      </UNDERSTANDING>
      
      <PERCEPTION>
        - This section describes the key visual findings or absence of findings in the image that are relevant to answering the question.
      </PERCEPTION>
      
      <INTERPRETATION>
        - This section links the perceived findings to their clinical meaning or implication.
      </INTERPRETATION>
      
      <ANSWER>
        - This section provides one concise and short answer to the question.
      </ANSWER>

    """

    medical_highlighter_system_role: str = "You are a medical expert tasked with highlighting the medically important terms of a given response to a medical question."
    def medical_highlighter_prompt(self, question: str, answer: str) -> str:
      return f"""Here is a medical question and a response to the question:
      Question: "{question}"
      Response: "{answer}"
      
      Highlight the terms that are medically important with respect to the Question by putting them inside <mask> and </mask> tags.
      Do not change any other part of the Response, only highlight the terms with the criteria.
      Only output the highlighted Response, do not include any other text.
      """
    

    def medical_text_only_refinement_prompt(self, question: str, answer: str, output: str, correctness: bool) -> str:
      return f"""Here is a medical question, the correct answer, and a model's given answer to the question:
Question: "{question}"
Correct Answer: "{answer}"
Given Answer by the model: "{output}"

The given answer is {"correct" if correctness else "incorrect"} based on the correct answer. Give me the {"incorrect" if correctness else "correct"} answer by changing the given answer as little as possible. Your response should only include the changed answer, do not include any other text.
"""
    def qwen_system_message(self) -> str:
      return self.qwen_system_role

# Global instance
prompt_templates = PromptTemplates()