from langchain_core.tools import tool
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "./models/qwen_reply_model"
)

model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen_reply_model",
    torch_dtype = torch.float16,
    device_map = "auto"
)

@tool
def reply(
    subject: str,
    email_text: str,
    tone: str = "professional"):
    """
    Generate a professional email reply based on the subject and email content
    """
    messages = [
        {
            "role":"system",
            "content": "You are an email assistant."
        },
        {
            "role":"user",
            "content": f"""
        Subject:
        {subject}
        
        Email:
        {email_text}
        
        Draft a {tone} reply."""
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens = 150,
        temperature = 0.8,
        do_sample=True,
        pad_token_id = tokenizer.eos_token_id
    )

    generated_ids = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return response

    
