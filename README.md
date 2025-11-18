# UBC CPSC AI Teaching Assistant

`https://huggingface.co/spaces/Samecho/ubc-cpsc-ai-demo`

## Project Goal

The base Llama 3 model has strong general knowledge, but it knows nothing about the specific, internal, and up-to-date course information of UBC (University of British Columbia) Computer Science (CPSC) — such as prerequisites or course descriptions.

Through QLoRA fine-tuning, this project personalizes Llama 3 into a specialized Q&A assistant for UBC CPSC courses using 247 lines of data.

## Tech Stack

Model: Meta Llama 3 8B Instruct

Fine-tuning Method: QLoRA (via the PEFT library)

Core Frameworks: PyTorch, Hugging Face transformers, trl

Hardware Optimization: bitsandbytes (4-bit quantization), sdpa (T4-compatible)

## How to Run (on Colab T4)

Clone this repository.

Set up HF_TOKEN in Colab Secrets.

Ensure the runtime is set to T4 GPU.

Install dependencies: ```pip install -r requirements.txt```

Run train.py.
