import os
import torch
import sys
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --------------------------------------------------
# RUN FROM /
# --------------------------------------------------
checkpoint_path = "data/models/checkpoint/prefix-tuning_Mistral-7B/checkpoint-4000"
base_model_name = "mistralai/Mistral-7B-v0.1"
input_dir = "data/cleaned_datasets/dataset_lapresse"
output_dir = "data/generated_summaries_fine-tuned_models/LORA-tuning_Mistral-7B_summaries"
batch_size = 16

os.makedirs(output_dir, exist_ok=True)

# Load the base model
print(f"Loading base model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Load the prefix-tuning checkpoint
print(f"Loading prefix-tuning checkpoint from: {checkpoint_path}")
model = PeftModel.from_pretrained(base_model, checkpoint_path)

# Load the tokenizer
# If you saved the tokenizer inside your prefix-tuning checkpoint,
# you can do:
#   tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
# Otherwise, load from the base model or wherever your tokenizer is stored:
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Define the prompt template
def build_prompt(filename, initial_text):
    return f"<titre>: {filename}\n<texte>: {initial_text}\n<résumé>: "

# Get list of all .txt files in the input directory
all_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

# Filter out already processed files
all_files = [f for f in all_files if not os.path.exists(os.path.join(output_dir, f))]

print(f"Running inference on {len(all_files)} files in {input_dir}")

# Process files in batches
for i in tqdm(range(0, len(all_files), batch_size), desc="Processing batches"):
    batch_files = all_files[i:i + batch_size]
    batch_prompts = []
    batch_filenames = []

    # Read and prepare batch
    for filename in batch_files:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            initial_text = f.read()

        batch_prompts.append(build_prompt(filename, initial_text))
        batch_filenames.append(filename)

    # Tokenize batch
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move tensors to GPU

    # Generate summaries
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=516)

    # Decode batch outputs
    batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Save summaries
    for filename, summary in zip(batch_filenames, batch_summaries):
        output_file_path = os.path.join(output_dir, filename)
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(summary)

print(f"Inference completed. Summaries saved in {output_dir}")