import os
import torch
import sys

from peft import (
    PrefixTuningConfig,
    get_peft_model
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import get_datasets

def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

# --------------------------------------------------
# RUN FROM /
# --------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
CHECKPOINT_SAVE = "data/models/checkpoint/LORA-tuning_Mistral-7B"
MODEL_SAVE = "data/models/LORA-fine-tuned_Mistral-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "cuda"},
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=33,
    prefix_projection=True,
)
model = get_peft_model(model, prefix_config)

# print_trainable_parameters(model)

def generate_and_tokenize_prompt(filename, initial_text, summary):
    prompt = f"<titre>: {filename}\n<texte>: {initial_text}\n<résumé>: "
    summary = f"{summary}{tokenizer.eos_token}"
    
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=2300, add_special_tokens=False)
    tokenized_summary = tokenizer(summary, truncation=True, max_length=1024, add_special_tokens=False)
    
    input_ids = tokenized_prompt.input_ids + tokenized_summary.input_ids
    labels = [-100] * len(tokenized_prompt.input_ids) + tokenized_summary.input_ids
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': [1] * len(input_ids)
    }

data_path = "data/cleaned_datasets/dataset_lapresse"
summaries_path = "data/summaries_datasets/mistral-summaries"

train_dataset, val_dataset, test_dataset = get_datasets(data_path, summaries_path)
print(len(train_dataset))

train_dataset = train_dataset.map(generate_and_tokenize_prompt)
val_dataset = val_dataset.map(generate_and_tokenize_prompt)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=3,
    num_train_epochs=3,
    learning_rate=1.4253725481594853e-05,
    fp16=True,
    save_total_limit=3,
    logging_steps=20,
    output_dir="models/checkpoint/LORA-tuning_Mistral-7B",
    optim="paged_adamw_8bit",
    lr_scheduler_type="polynomial",
    warmup_ratio=0.024239631853640843,
    report_to="tensorboard",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model.config.use_cache = False

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)

trainer.train()

os.makedirs("models", exist_ok=True)
model.save_pretrained(MODEL_SAVE)
tokenizer.save_pretrained(MODEL_SAVE)