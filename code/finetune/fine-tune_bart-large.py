import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from peft import LoraConfig, PrefixTuningConfig, get_peft_model, TaskType
from torch.optim import AdamW

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define prompt template
PROMPT_TEMPLATE = "<texte>:{}\n\n<Résumé>:"

# Define chunking function
def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]

# Define dataset class
class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def process_text(self, text):
        chunks = chunk_text(text)
        summarized_chunks = []
        for chunk in chunks:
            input_text = PROMPT_TEMPLATE.format(tokenizer.decode(chunk))
            input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(device)
            summary_ids = model.generate(input_ids=input_ids, max_new_tokens=128, num_beams=5, early_stopping=True, forced_bos_token_id=tokenizer.bos_token_id)
            summarized_chunks.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        
        # Summarize the concatenated text
        concatenated_text = " ".join(summarized_chunks)
        final_input_text = PROMPT_TEMPLATE.format(concatenated_text)
        final_input_ids = tokenizer(final_input_text, return_tensors="pt", truncation=True).input_ids.to(device)
        final_summary_ids = model.generate(input_ids=final_input_ids, max_new_tokens=256, num_beams=5, early_stopping=True, forced_bos_token_id=tokenizer.bos_token_id)
        
        return concatenated_text, tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
    
    def __getitem__(self, idx):
        orig, summ = self.data[idx]
        processed_input, processed_summary = self.process_text(orig)
        return processed_input, processed_summary

# Load dataset object
dataset_path = "data/summarization_dataset.pt"
torch.serialization.add_safe_globals([SummarizationDataset])
data = torch.load(dataset_path)
dataset = SummarizationDataset(data, tokenizer)

# Initialize DataLoader
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Apply LoRA tuning
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# Apply Adapter tuning
adapter_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=20,  # Number of virtual tokens
)
model = get_peft_model(model, adapter_config)

# Define optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
# Training loop with real-time tracking
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs = tokenizer(list(inputs), return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids.to(device)
        targets = tokenizer(list(targets), return_tensors="pt", padding=True, truncation=True, max_length=256).input_ids.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} complete. Avg Loss: {total_loss / len(dataloader):.4f}")


# Save the fine-tuned model
output_dir = "data/models/fine_tuned_bart"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Fine-tuning complete. Model saved.")