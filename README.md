# ai-therapist

# 🧠 AI Therapist

An AI-powered conversational mental wellness assistant built using **TinyLlama**, **LoRA fine-tuning**, and **Streamlit**. The model was fine-tuned on therapist-client conversations from the CounselChat dataset to provide supportive and empathetic responses in a conversational interface.

## Features

* Fine-tuned TinyLlama-1.1B using LoRA (Low-Rank Adaptation)
* Trained on real therapist-client conversations from the CounselChat dataset
* Conversational chat interface built with Streamlit
* Multi-turn conversation memory
* 4-bit quantized training for resource-efficient fine-tuning
* Optimized for low-VRAM GPUs (RTX 3050 4GB)
* Model merging for simplified deployment and inference

## Project Architecture

CounselChat Dataset
↓
Data Cleaning & Preprocessing
↓
Instruction Dataset Creation
↓
LoRA Fine-Tuning (TinyLlama-1.1B)
↓
Model Merging
↓
Streamlit Web Application

## Tech Stack

### Machine Learning

* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* TRL (Transformer Reinforcement Learning Library)
* BitsAndBytes Quantization

### Deployment

* Streamlit

### Data Processing

* Pandas
* Datasets Library

## Training Details

| Component            | Value                     |
| -------------------- | ------------------------- |
| Base Model           | TinyLlama-1.1B-Chat-v1.0  |
| Fine-Tuning Method   | LoRA                      |
| Dataset              | CounselChat               |
| Samples              | 1,371                     |
| Quantization         | 4-bit                     |
| GPU                  | RTX 3050 Laptop GPU (4GB) |
| Training Epochs      | 1                         |
| Trainable Parameters | ~1.1 Million              |

## Results

The model was successfully fine-tuned to generate supportive conversational responses while maintaining a lightweight deployment footprint. The project demonstrates:

* Instruction tuning
* Parameter-efficient fine-tuning (PEFT)
* Quantized LLM training
* Conversational AI deployment
* End-to-end ML workflow from data preparation to production interface

## Screenshots

### Chat Interface

![Chat Interface](screenshots/Screenshot%20(94).png)

### Example Conversation

![example](screenshots/Screenshot%20(95).png)

## Installation

```bash
git clone https://github.com/Soyam-Patra/ai-therapist.git
cd ai-therapist

pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

## Reproducing Training

```bash
python train.py
```

## Model Weights

Model weights are not included in this repository due to GitHub file size limitations.

To reproduce the model:

1. Run the training pipeline using `train.py`
2. Merge LoRA adapters using `merge.py`
3. Launch the application using `app.py`

## Skills Demonstrated

* Large Language Models (LLMs)
* LoRA Fine-Tuning
* Hugging Face Ecosystem
* Quantization Techniques
* Conversational AI
* Streamlit Deployment
* GPU Optimization
* Data Preprocessing
* Prompt Engineering
* End-to-End ML Development

## Author

**Soyam Patra**

GitHub: https://github.com/Soyam-Patra
LinkedIn: https://linkedin.com/in/soyam-patra
