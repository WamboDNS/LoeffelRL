from transformers import AutoModelForCausalLM
from transformers.utils import logging
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

load_dotenv()
logging.set_verbosity_info() 
token = os.getenv("HF_TOKEN")
login(token=token)

print("Get base model")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16)

print("Get adapter")
peft_model = PeftModel.from_pretrained(base_model, "wambosec/Qwen2.5-7B-Instruct-SFT-spoon-language-lora")

print("Merge model")
merged_model = peft_model.merge_and_unload()

# ⚠️ Important: Cast to float16
merged_model.half()

print("Save model (fp16)")
merged_model.save_pretrained("Qwen2.5-7B-Instruct-spoon-language-SFT-16bit", safe_serialization=True)

print("✅ Done. Saved as float16.")
