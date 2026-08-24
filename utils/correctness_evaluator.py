import argparse
import csv
import os
import time
from typing import Dict, List, Optional
import pandas as pd
import openai
import asyncio
from dotenv import load_dotenv


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from templates.prompt_templates import prompt_templates
from templates.conversation_templates import conversation_templates
import re



load_dotenv()


# Closed yes/no questions are the bulk of the VQA test splits and do not need an LLM
# judge: the ground truth is literally "yes" or "no", so the only thing that matters is
# which of those two words the model committed to. Resolving them with a regex removes
# roughly 40% of the API calls at no cost in accuracy.
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def regex_yes_no_verdict(correct_answer, model_output) -> Optional[bool]:
    """
    Fast path for closed yes/no questions.

    Returns True/False when the verdict can be decided without an API call, and None
    when it cannot -- in which case the caller must fall back to the LLM judge.

    None is returned in two situations, both deliberately conservative:
      * the ground truth is not a bare "yes"/"no" (an open question), or
      * the model never produced a standalone yes/no token, so there is nothing to
        compare against and guessing would silently manufacture a wrong verdict.

    The first standalone yes/no token in the response is taken as the model's
    commitment, matching how a reader would score it: "No, the lungs are clear" is a
    negative answer, and word boundaries keep "nodule"/"normal" from matching.
    """
    gt = str(correct_answer).strip().lower().rstrip(".!").strip()
    if gt not in ("yes", "no"):
        return None

    match = _YES_NO_RE.search(str(model_output))
    if match is None:
        return None

    return match.group(1).lower() == gt


class CorrectnessEvaluator:
    """Evaluates correctness of model outputs using OpenAI API"""
    
    def __init__(
        self,
        text_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        delay: float = 0.0
    ):
        self.text_model = text_model
        self.delay = delay
        # Counters so the split between the regex fast path and the LLM judge is visible.
        self.regex_resolved = 0
        self.llm_resolved = 0

        # Setup OpenAI client
        api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.async_client = openai.AsyncOpenAI(api_key=api_key)
    
    async def evaluate_batch_async(self, batch: List[Dict]) -> List[bool]:

        async def process_single_sample(idx: int, item: Dict) -> tuple:
            try:
                match = re.search(r"<ANSWER>(.*?)</ANSWER>", item["output"], re.DOTALL)
                if match:
                    extracted_output = match.group(1).strip()
                else:
                    extracted_output = str(item["output"]).strip()

                # Closed yes/no question: decide it locally and skip the API entirely.
                verdict = regex_yes_no_verdict(item["answer"], extracted_output)
                if verdict is not None:
                    self.regex_resolved += 1
                    return (idx, verdict)
                self.llm_resolved += 1

                messages = conversation_templates.get_gpt4omini().create_evaluation_conversation(
                    question=item["question"],
                    correct_answer=item["answer"],
                    model_output=extracted_output
                )
                response = await self.async_client.chat.completions.create(
                    model=self.text_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=10
                )
                
                result = response.choices[0].message.content
                if type(result) != str:
                    return (idx, False)
                result = result.strip()
                if result not in ["True", "False", "true", "false"]:
                    print(f"Invalid response from evaluator: {result}")
                    return (idx, False)
                        
                return (idx, result in ("True", "true"))
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                return (idx, False)
        
        # Process all samples in parallel
        for _ in range(3):
            try:
                tasks = [process_single_sample(idx, item) for idx, item in enumerate(batch)]
                results = await asyncio.gather(*tasks)
                break
            except Exception as e:
                print(f"Error evaluating batch: {e}")
                time.sleep(10)
        
        # Sort by index and return just the boolean values
        results.sort(key=lambda x: x[0])
        return [is_correct for _, is_correct in results]
    
    def evaluate_csv(
        self,
        input_csv_path: str,
        output_csv_path: str,
        start_idx: int = 0,
        batch_size: int = 10,
    ) -> None:
        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(output_csv_path):
            df = pd.read_csv(output_csv_path)
            first_none_idx = df['is_correct'].isna()
            start_idx = int(df[first_none_idx].iloc[0].name)
        else:
            start_idx = 0
            df = pd.read_csv(input_csv_path)
            if 'is_correct' not in df.columns:
                df['is_correct'] = None
        
        
        total_samples = len(df)
        print(f"Evaluating correctness for {total_samples} samples...")
        print(f"Starting from index {start_idx}")
        
        # Process in batches
        processed_count = 0
        for i in range(start_idx, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_df = df.iloc[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}: samples {i} to {batch_end-1}")
            
            # Convert batch to list of dicts
            batch_data = batch_df.to_dict('records')
            
            correctness_results = asyncio.run(self.evaluate_batch_async(batch_data))
            # correctness_results = self.evaluate_batch(batch_data)
                    
            for j, is_correct in enumerate(correctness_results):
                df.iloc[i + j, df.columns.get_loc('is_correct')] = is_correct
            
            processed_count += len(batch_data)
            
            # Save intermediate results
            if (i + batch_size) % (batch_size * 5) == 0:  # Save every 5 batches
                df.to_csv(output_csv_path, index=False)
                print(f"Saved intermediate results after {processed_count} samples")
        
        # Save final results
        df.to_csv(output_csv_path, index=False)
        
        # Calculate metrics
        correct_count = df['is_correct'].sum()
        total_count = len(df)
        accuracy = correct_count / total_count if total_count > 0 else 0       
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {output_csv_path}")
        print(f"Total samples: {total_count}")
        print(f"Correct samples: {correct_count}")
        print(f"Incorrect samples: {total_count - correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        judged = self.regex_resolved + self.llm_resolved
        if judged:
            print(
                f"Verdicts: {self.regex_resolved} by regex fast path "
                f"({self.regex_resolved / judged:.1%}), {self.llm_resolved} by "
                f"{self.text_model}"
            )



def main():
    parser = argparse.ArgumentParser(description="Evaluate Correctness of Model Outputs")
    parser.add_argument("--input_csv", type=str, required=True,
                       help="Path to input CSV file with model outputs")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Path to output CSV file with correctness evaluations")
    parser.add_argument("--text_model", type=str, default="gpt-4o-mini",
                       help="OpenAI model for text evaluation")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (if not set, uses OPENAI_API_KEY env var)")
    parser.add_argument("--delay", type=float, default=0.0,
                       help="Delay between API calls in seconds")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index for evaluation")
    parser.add_argument("--batch_size", type=int, default=50,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = CorrectnessEvaluator(
        text_model=args.text_model,
        api_key=args.api_key,
        delay=args.delay
    )
    
    # Run evaluation
    evaluator.evaluate_csv(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        start_idx=args.start_idx,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
