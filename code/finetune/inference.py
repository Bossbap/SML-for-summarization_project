import os
import torch
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the parent directory to import the dataset utility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import get_datasets

# Path to the fine-tuned model
model_save_path = "models/fine_tuned_llama3.2-3B_V1"

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_save_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
tokenizer.pad_token = tokenizer.eos_token

# Define the prompt template for inference (without summary input)
def build_prompt(filename, initial_text):
    return f"<titre>: {filename}\n<texte>: {initial_text}\n<résumé>: "

# Load datasets
data_path = "data/cleaned_lapresse_dataset"
summaries_path = "data/generated_summaries_lapresse"
# get_datasets is assumed to return train, val, and test datasets.
_, _, test_dataset = get_datasets(data_path, summaries_path)

# Run inference on the test dataset
print("Running inference on test dataset...\n")
for idx, data in enumerate(test_dataset):
    # Assuming each data point is a tuple: (filename, initial_text, summary)
    filename, initial_text, _ = data
    prompt = build_prompt(filename, initial_text)
    
    # Tokenize and move input to model's device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    # Generate a summary (adjust max_new_tokens as needed)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=516)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test Example {idx}:")
    print(generated_text)
    print("=" * 80)