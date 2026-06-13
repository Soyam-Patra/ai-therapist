import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA:", torch.cuda.is_available())
print("BF16:", torch.cuda.is_bf16_supported())
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Applying LoRA...")

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.print_trainable_parameters()

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files="therapist_sft.json",
    split="train"
)

MAX_LENGTH = 256

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

dataset = dataset.map(tokenize)

print(dataset)

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

torch.cuda.empty_cache()
print("Starting training...")

trainer.train()

print("Saving model...")

model.save_pretrained("./therapist_lora")
tokenizer.save_pretrained("./therapist_lora")

print("Done!")