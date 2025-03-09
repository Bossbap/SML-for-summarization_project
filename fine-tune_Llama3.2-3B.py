import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, PrefixTuningConfig, get_peft_model, TaskType

# ✅ Define model name
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# ✅ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# ✅ Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Define LoRA tuning configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# Apply Adapter tuning
adapter_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=20,  # Number of virtual tokens
)
model = get_peft_model(model, adapter_config)

# ✅ Load dataset (modify path if needed)
dataset_path = "data/summarization_dataset.pt"
torch.serialization.add_safe_globals([])
data = torch.load(dataset_path, weights_only=False)

# ✅ Convert dataset to Hugging Face format
class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        orig, summ = self.data[idx]
        prompt = f"<texte>: {orig}\n\n<Résumé>:"
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)["input_ids"].squeeze()
        label_ids = self.tokenizer(summ, return_tensors="pt", truncation=True, max_length=512)["input_ids"].squeeze()
        return {"input_ids": input_ids, "labels": label_ids}

dataset = SummarizationDataset(data, tokenizer)

# ✅ Define training arguments
training_args = TrainingArguments(
    output_dir="data/models/fine_tuned_llama3",
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=50,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Mixed precision
    push_to_hub=False
)

# ✅ Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# ✅ Start training
trainer.train()

# ✅ Save fine-tuned model
trainer.save_model("data/models/fine_tuned_llama3")
tokenizer.save_pretrained("data/models/fine_tuned_llama3")

print("Fine-tuning complete. Model saved.")