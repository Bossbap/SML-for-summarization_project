import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
INPUT_FOLDER = "data/cleaned_datasets/dataset_lapresse"
OUTPUT_FOLDER = "results/baseline/summaries_lapresse"
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)

def generate_summary(text):
    prompt = f"Résumes le texte suivant:\n\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=512, num_beams=5)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Iterate over text files
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".txt"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        
        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
        
        summary = generate_summary(text)
        
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(summary)
        
        print(f"Processed: {filename}")

print("Baseline summaries generated successfully.")