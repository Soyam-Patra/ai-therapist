import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "./therapist_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

prompt = """
### Instruction:
I feel anxious all the time and can't focus on my studies.

### Response:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)