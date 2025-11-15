import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"

print("Installing libraries...")
# !pip install -q transformers peft accelerate bitsandbytes trl datasets torch
print("Libraries installed.")

import torch
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Logging into Hugging Face...")
try:
    token = userdata.get('HF_TOKEN')
    login(token=token)
    print("Login successful.")
except:
    print("Login failed. Please check your HF_TOKEN.")
    pass

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_id = "mlabonne/guanaco-llama2-1k"

print("Loading model configuration...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)
print("Model loaded.")

from datasets import load_dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="ubc_courses.jsonl", split="train")
print(f"Dataset loaded: {len(dataset)} samples.")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("Preparing model for LoRA...")
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
print("LoRA configuration applied.")

from trl import SFTConfig, SFTTrainer

print("Initializing trainer...")
training_args = SFTConfig(
    output_dir="./my_llama3_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="no",
    fp16=False,
    bf16=True,
    packing=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)
print("Trainer initialized.")

print("Starting training...")
trainer.train()
print("Training completed.")

adapter_name = "my-llama3-8b-adapter"
print("Saving adapter...")
trainer.save_model(adapter_name)
print("Adapter saved.")
