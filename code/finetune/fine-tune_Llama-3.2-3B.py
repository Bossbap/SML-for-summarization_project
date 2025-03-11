import os
import torch
import sys

from peft import (
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import get_datasets


MODEL_NAME = "meta-llama/Llama-3.2-3B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

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

model = get_peft_model(model, config)

adapter_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=40,
    prefix_projection=True,
)
model = get_peft_model(model, adapter_config)

def generate_and_tokenize_prompt(filename, initial_text, summary):
    prompt = f"<titre>: {filename}\n<texte>: {initial_text}\n<résumé>: "
    summary = f"{summary}{tokenizer.eos_token}"
    
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=4096, add_special_tokens=False)
    tokenized_summary = tokenizer(summary, truncation=True, max_length=1024, add_special_tokens=False)
    
    input_ids = tokenized_prompt.input_ids + tokenized_summary.input_ids
    labels = [-100] * len(tokenized_prompt.input_ids) + tokenized_summary.input_ids
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': [1] * len(input_ids)
    }

data_path = "data/cleaned_lapresse_dataset"
summaries_path = "data/generated_summaries_lapresse"

train_dataset, val_dataset, test_dataset = get_datasets(data_path, summaries_path)
print(len(train_dataset))

train_dataset = train_dataset.map(generate_and_tokenize_prompt)
val_dataset = val_dataset.map(generate_and_tokenize_prompt)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=20,
    output_dir="output_dir",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="tensorboard",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

model.config.use_cache = False

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)

trainer.train()

model_save_path = "models/fine_tuned_llama3.2-3B_V1"
os.makedirs("models", exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)