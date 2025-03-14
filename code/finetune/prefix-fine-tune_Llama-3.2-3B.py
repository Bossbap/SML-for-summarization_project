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
MODEL_NAME = "meta-llama/Llama-3.2-3B"
CHECKPOINT_SAVE = "data/models/checkpoint/prefix-tuning_Llama-3.2-3B"
MODEL_SAVE = "data/models/prefix-fine-tuned_Llama-3.2-3B"

bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=52,
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
    gradient_accumulation_steps=6,
    num_train_epochs=3,
    learning_rate=0.0004299856458270593,
    bf16=True,
    save_total_limit=3,
    logging_steps=20,
    output_dir="models/checkpoint/LORA-tuning_Mistral-7B",
    optim="paged_adamw_8bit",
    lr_scheduler_type="linear",
    warmup_ratio=0.11290227454698583,
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