import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# --------------------------------------------------
# RUN FROM /
# --------------------------------------------------
INPUT_FOLDER = "data/cleaned_datasets/dataset_lapresse"
OUTPUT_FOLDER = "data/baseline_generated_summaries/baseline_mistral"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, quantization_config=bnb_config
).to(device)

def generate_summaries(prompts):
    """
    Given a list of prompts, tokenizes them with padding,
    generates summaries in batch, and returns the decoded outputs.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2300
    ).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, num_beams=1)
    
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

# Get a list of all .txt files in the input folder that haven't been processed
all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
pending_files = []
for filename in all_files:
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(output_path):
        pending_files.append(filename)
    else:
        print(f"Skipping {filename}, already processed.")

BATCH_SIZE = 16  # Adjust as needed based on your GPU memory and throughput

# Process files in batches
for i in tqdm(range(0, len(pending_files), BATCH_SIZE), desc="Processing batches", unit="batch"):
    batch_files = pending_files[i : i + BATCH_SIZE]
    prompts = []
    # Prepare prompts for each file in the batch
    for filename in batch_files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
        title = os.path.splitext(filename)[0]
        prompt = f"Résumes le texte suivant:\ntitre: {title}\n\ntexte: {text}"
        prompts.append(prompt)
    
    # Generate summaries for the batch
    summaries = generate_summaries(prompts)
    
    # Save each summary using the same filename
    for filename, summary in zip(batch_files, summaries):
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(summary)
        print(f"Processed: {filename}")

print("Baseline summaries generated successfully.")