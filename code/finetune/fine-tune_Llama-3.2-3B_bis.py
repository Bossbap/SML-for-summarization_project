import pandas as pd
import transformers
from datasets import Dataset
import os
import torch

from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import DataCollatorForSeq2Seq

MODEL_NAME = "meta-llama/Llama-3.2-3B"

device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16  # Explicitly setting dtype
).to(device)  # Ensure it is moved to CUDA

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    config
)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)  # This should print LoRA layers

os.environ["TOKENIZERS_PARALLELISM"] = "false"

data_path = "data/cleaned_datasets/dataset_lapresse"
summaries_path = "data/generated_summaries_lapresse"

# Fetch files
texts = []
summaries = []

for filename in os.listdir(data_path):
    text_file = os.path.join(data_path, filename)
    summary_file = os.path.join(summaries_path, filename)
    
    if os.path.exists(summary_file):
        with open(text_file, "r", encoding="utf-8") as tf, open(summary_file, "r", encoding="utf-8") as sf:
            texts.append(tf.read().strip())
            summaries.append(sf.read().strip())

# Create a pandas DataFrame
df = pd.DataFrame({"text": texts, "summary": summaries})

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

data = {"train": dataset}

def generate_prompt(data_point):
    return f"<texte>: {data_point['text']}\n<résumé>: {data_point['summary']}"

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)+tokenizer.eos_token 
    tokenized_full_prompt = tokenizer(full_prompt, return_tensors='pt')
    labels = tokenized_full_prompt.input_ids.clone()

    end_prompt_idx = full_prompt.find("<résumé>:")

    labels[:, :end_prompt_idx] = -100

    return {
        'input_ids': tokenized_full_prompt.input_ids.flatten(),
        'labels': labels,
        'attention_mask': tokenized_full_prompt.attention_mask.flatten(),
    }

data = data["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)

OUTPUT_DIR = "experiments"

# Training arguments
training_args = transformers.TrainingArguments(
    label_names=["labels"],  # Explicitly specify label names
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Reduced to prevent OOM
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,  # Use FP16 instead of BF16 if needed
    save_total_limit=3,
    logging_steps=20,
    output_dir="output_dir",
    max_steps=600,   # Try more steps if you can
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="tensorboard",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model.config.use_cache = False

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)

trainer.train()