import os
import json
import dspy
import datasets
from pydantic import BaseModel, Field
from typing import List
from tqdm import tqdm
import time

#CUDA_VISIBLE_DEVICES=3 vllm serve casperhansen/deepseek-r1-distill-qwen-32b-awq --max-model -len 32768 --gpu-memory-utilization 0.9 --quantization awq_marlin --port 8000

class IncentiveInstance(BaseModel):
    # using pydantic to ensure our outputs are the right data types
    instrument_text: str = Field(description="The verbatim text (word or phrase) referencing a policy incentive or instrument")
    class_text: str = Field(description="The verbatim text (word or phrase) indicating the class of the policy incentive or instrument")
    label: str = Field(description="The class of policy incentive or instrument")

class IncProc(dspy.Signature):
    """Given a sentence from a policy document, determine if it contains policy incentives or instruments. 
    If so, then for each instance, extract an item containing 
    1) the explicit reference to the policy incentive or instrument (OR 'None' if not explicit)
    2) the explicit text that indicates the class of policy incentive or instrument (OR 'None' if not explicit)
    3) the class of the policy incentive or instrument
    """
    text: str = dspy.InputField(desc="Sentence from a policy document to analyze.")
    output: List[IncentiveInstance] = dspy.OutputField(desc="""Given a sentence from a policy document, 
        FIRST 
        Determine if it mentions a type of policy action, instrument, or incentive.
        - If not, return an empty list. 
        - If it does mention at least one type, make sure the mention is not just a definition of what an incentive type is or an example provided as context for a different discussion. 
            Future tense is usually a good indicator that a mention is relevant. 
        THEN 
        FOR EACH RELEVANT INCENTIVE IN THE SENTENCE                                          
        - extract the text (word or short phrase) that explicitly references or names the incentive (e.g. the title of a scheme) [only if present, this element is not mandatory]
        - extract the text (word or short phrase) that indicates what class of incentive the reference is (e.g. "equipment" indicates the class Supplies) [only if present, this element is not mandatory]
        - classify the incentive type with one of the below labels:
            - Credit: The provision of repayable finance such as loans or insurance to encourage an action.
            - Direct_payment: The provision of non-repayable finance such as cash or grants to encourage an action.
            - Fine: The establishment of a financial penalty for performing a prohibited action or not performing a required action.
            - Supplies: The provision of physical goods, material support, or equipment to encourage an action.
            - Tax_deduction: The reduction of tax liability to encourage an action.
            - Technical_assistance: The provision of trainings, expertise, or advisory services to encourage an action.
        THEN
        Return a list containing the information for each incentive in the sentence
    """)

def main():
    cwd = os.getcwd()

    vllm_model = dspy.LM(
        api_base="http://localhost:8000/v1",
        api_key="local",
        model="openai/casperhansen/deepseek-r1-distill-qwen-32b-awq",
        max_tokens=4096
    )
    dspy.configure(lm=vllm_model)

    ds = datasets.load_dataset("mawaskow/irish_forestry_incentives")

    program = dspy.ChainOfThought(IncProc)

    results = []
    st = time.time()
    for entry in tqdm(ds['train']):
        pt = time.time()
        try:
            pred = program(text=entry['text'])
            if isinstance(pred.output, list):
                pyd_output = [item.model_dump() for item in pred.output] #item.dict() if pydantic is v1
            else:
                pyd_output = pred.output.model_dump() if pred.output else []
            reasoning_val = getattr(pred, 'reasoning', "N/A")
        except Exception as e:
            print(f"\n[ERROR] Failed to parse item: {entry['text'][:50]}... Error: {e}")
            pyd_output = []
            reasoning_val = "Parsing failed"
        results.append({
            "sentence": entry['text'],
            "label": entry['label'],
            "output": pyd_output,
            "reasoning": reasoning_val
        })
        print(f"\t--- Example processed in {round((time.time()-pt)/60,2)} min ---")
        # periodic saving
        if len(results) % 10 == 0:
            with open(f"{cwd}/data/incentive_results_partial.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
    print(f"\tProgram completed in {round((time.time()-st)/3600,2)} hr ({round((time.time()-st)/60,2)} min) ---")
    with open(f"{cwd}/data/incentive_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()