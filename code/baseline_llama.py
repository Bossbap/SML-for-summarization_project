import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Paths
INPUT_FOLDER = "data/cleaned_lapresse_dataset"
OUTPUT_FOLDER = "data/baseline"
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)

def generate_summary(title, text):
    prompt = f"Résumes le texte suivant:\ntitre: {title}\n\ntexte: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2300).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=512, num_beams=3)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Iterate over text files
files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]

for filename in tqdm(files, desc="Processing files", unit="file"):
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    
    # Skip processing if the file already exists in the output folder
    if os.path.exists(output_path):
        print(f"Skipping {filename}, already processed.")
        continue
    
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read().strip()
    
    title = os.path.splitext(filename)[0]
    summary = generate_summary(title, text)
    
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(summary)
    
    print(f"Processed: {filename}")

print("Baseline summaries generated successfully.")