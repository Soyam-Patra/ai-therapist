import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./therapist_merged"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model

tokenizer, model = load_model()

st.title("AI Therapist")

user_input = st.text_area(
    "How are you feeling today?"
)

if st.button("Send"):

    prompt = f"""
### Instruction:
{user_input}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "### Response:" in response:
        response = response.split(
            "### Response:"
        )[-1].strip()

    st.write(response)