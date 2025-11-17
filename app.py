import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

print(">>> Starting to load model and tokenizer...")

MODEL_ID = "Samecho/ubc-cpsc-ai-llama3-8b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
print(">>> Model loading successful!")

def chat_function(message, chat_history):
    print(f">>> Received message: {message}")
    
    messages = []
    for user_msg, ai_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print(">>> Generating response (streaming)...")
    
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

print(">>> Launching Gradio Demo...")

example_prompts = [
    "What are the prerequisites for CPSC 221?",
    "Tell me about CPSC 210.",
    "What is CPSC 310 'Software Engineering' about?"
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 UBC CPSC AI Assistant (Demo)
        
        This project demonstrates a Llama 3 8B-Instruct model fine-tuned using QLoRA on a free Google Colab T4 GPU.
        
        **Base Model (Llama 3):** Has no specific knowledge of UBC CPSC courses and will typically apologize or refuse to answer.
        **This Model (Mine):** Was fine-tuned on 247 data points scraped from the UBC CPSC website. It can now answer specific questions about course descriptions and prerequisites.
        
        **GitHub Repo:** [github.com/Samecho/ubc-cpsc-ai](https://github.com/Samecho/ubc-cpsc-ai)
        """
    )
    
    gr.ChatInterface(
        fn=chat_function,
        title="UBC CPSC AI Assistant",
        chatbot=gr.Chatbot(height=500),
        examples=example_prompts,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()