from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load Hugging Face token
load_dotenv()
token = os.getenv("HF_TOKEN")
login(token=token)

# Model path (local) and repo name (on Hub)
local_model_path = "Qwen2.5-7B-Instruct-spoon-language-SFT-16bit"
repo_name = "wambosec/Qwen2.5-7B-Instruct-spoon-language-SFT"

# Load and push the merged model
print("Loading merged model...")
model = AutoModelForCausalLM.from_pretrained(local_model_path)
print("Pushing model to Hugging Face Hub...")
model.push_to_hub(repo_name)

# Optional: push tokenizer (you can also load and customize if needed)
print("Pushing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.push_to_hub(repo_name)

print("✅ Upload complete.")
