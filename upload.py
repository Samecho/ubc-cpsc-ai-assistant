import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from google.colab import userdata

print("Logging into Hugging Face...")
try:
    token = userdata.get('HF_TOKEN')
    login(token=token)
    print("✅ Login successful!")
except Exception as e:
    print(f"Login failed: {e}")

new_adapter_id = "Samecho/ubc-cpsc-ai-llama3-8b"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print("⏳ Loading 4-bit Base Model (for merging)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}, 
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

print("Base Model loaded successfully!")

print("Attaching LoRA adapter (./my-llama3-8b-adapter)...")

model = PeftModel.from_pretrained(base_model, "./my-llama3-8b-adapter")

print("Merging model...")

model = model.merge_and_unload()

print("Model merge complete!")

print(f"Full model (16GB) is being uploaded to: {new_adapter_id} ... (This will be very, very slow, please be patient)")

model.push_to_hub(new_adapter_id, use_auth_token=True, safe_serialization=True)

tokenizer.push_to_hub(new_adapter_id, use_auth_token=True)

print(f"Congratulations! Your personalized AI has been published at:")
print(f"https://huggingface.co/{new_adapter_id}")
print("You can now proceed to the final step (Gradio Demo)!")