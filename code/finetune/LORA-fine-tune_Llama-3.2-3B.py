import os
import torch
import sys

from peft import (
    LoraConfig,
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

# --------------------------------------------------
# RUN FROM /
# --------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B"
CHECKPOINT_SAVE = "data/models/checkpoint/LORA-tuning_Llama-3.2-3B"
MODEL_SAVE = "data/models/LORA-fine-tuned_Llama-3.2-3B"

def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 6.0,
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

# Apply LoRA
config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.13591984779369298,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

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
train_dataset = train_dataset.map(generate_and_tokenize_prompt)
val_dataset = val_dataset.map(generate_and_tokenize_prompt)

# print('='*100)
# print(len(train_dataset[1]))
# print(train_dataset[1])
# print('\n\n')
# print('='*100)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=0.00018785317317224597,
    fp16=True,
    save_total_limit=5,
    logging_steps=20,
    output_dir=CHECKPOINT_SAVE,
    optim="paged_adamw_8bit",
    lr_scheduler_type="polynomial",
    warmup_ratio=0.02343885260014309,
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

# Save final model
os.makedirs("models", exist_ok=True)
model.save_pretrained(MODEL_SAVE)
tokenizer.save_pretrained(MODEL_SAVE)