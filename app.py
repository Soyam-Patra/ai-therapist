import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./therapist_merged"

SYSTEM_PROMPT = ""
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

st.title("🧠 AI Therapist")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("How are you feeling today?")

if user_input:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Build conversation context
    conversation = SYSTEM_PROMPT + "\n\n"

    for msg in st.session_state.messages[-6:]:

        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"

        else:
            conversation += f"Therapist: {msg['content']}\n"

    conversation += "Therapist:"

    inputs = tokenizer(
        conversation,
        return_tensors="pt"
    ).to(model.device)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            response = response[len(conversation):].strip()

            # Fallback cleanup
            if "User:" in response:
                response = response.split("User:")[0].strip()

            st.write(response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )