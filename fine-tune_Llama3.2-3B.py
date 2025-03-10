import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, PrefixTuningConfig, get_peft_model, TaskType

# ✅ Define model name
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# ✅ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.config.use_cache = False

# ✅ Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Define LoRA tuning configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

# Apply Adapter tuning
adapter_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=40,
    token_dim=4096,
    prefix_projection=True,
)
model = get_peft_model(model, adapter_config)

# ✅ Load dataset
class SummarizationDataset(Dataset):
    def __init__(self, articles_dir, summaries_dir, tokenizer, max_input_length=4096, max_new_tokens=512):
        self.articles_dir = articles_dir
        self.summaries_dir = summaries_dir
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.file_names = [f for f in os.listdir(articles_dir) if f.endswith(".txt")]

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        article_path = os.path.join(self.articles_dir, file_name)
        summary_path = os.path.join(self.summaries_dir, file_name)

        with open(article_path, "r", encoding="utf-8") as f:
            article_text = f.read()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()

        prompt = f"<texte>: {article_text}\n\n<Résumé>:"

        input_enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
            padding="max_length"
        )

        label_enc = self.tokenizer(
            summary_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_output_length,
            padding="max_length"
        )

        # Replace pad tokens in labels with -100 for loss masking
        label_enc["input_ids"][label_enc["input_ids"] == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": label_enc["input_ids"].squeeze(0),
        }

# Define paths to dataset
dataset_path_articles = "data/cleaned_datasets/dataset_lapresse"
dataset_path_summaries = "data/generated_summaries_lapresse"

# Initialize dataset and dataloader
dataset = SummarizationDataset(dataset_path_articles, dataset_path_summaries, tokenizer)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding="longest",  # Ensures all batch inputs match in size
    label_pad_token_id=-100
)

dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=data_collator)

# ✅ Define training arguments
training_args = TrainingArguments(
    output_dir="data/models/fine_tuned_llama3",
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=50,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Mixed precision
    push_to_hub=False,
    label_names=["labels"]  # Ensures Trainer recognizes label field
)

# ✅ Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1)  # Improves efficiency
)

# ✅ Start training
trainer.train()

# ✅ Save fine-tuned model
trainer.save_model("data/models/fine_tuned_llama3")
tokenizer.save_pretrained("data/models/fine_tuned_llama3")

print("Fine-tuning complete. Model saved.")