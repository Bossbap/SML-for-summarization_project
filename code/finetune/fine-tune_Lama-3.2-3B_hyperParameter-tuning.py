import os
import torch
import sys
import optuna
import time
import json

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
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)

# Add the parent directory so we can import get_datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import get_datasets

MODEL_NAME = "meta-llama/Llama-3.2-3B"
data_path = "data/cleaned_lapresse_dataset"
summaries_path = "data/generated_summaries_lapresse"

def objective(trial: optuna.trial.Trial):
    # ----------------------------
    # Sample hyperparameters
    # ----------------------------
    # LoRA configuration
    lora_rank = trial.suggest_categorical("lora_rank", [16, 32, 64])
    lora_alpha = 2 * lora_rank
    lora_dropout = trial.suggest_float("lora_dropout", 0.01, 0.2)
    
    # Prefix Tuning configuration
    num_virtual_tokens = trial.suggest_int("num_virtual_tokens", 20, 100)
    
    # TrainingArguments configuration
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 4, 16, step=4)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["cosine", "linear", "polynomial"])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Print the hyperparameter combination for this trial
    print("Trial parameters:")
    print(f"  lora_rank: {lora_rank}, lora_alpha: {lora_alpha}, lora_dropout: {lora_dropout}")
    print(f"  num_virtual_tokens: {num_virtual_tokens}")
    print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  lr_scheduler_type: {lr_scheduler_type}")
    print(f"  warmup_ratio: {warmup_ratio}")
    
    # ----------------------------
    # Initialize model and tokenizer for this trial
    # ----------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_8bit = True,
        llm_int8_threshold = 6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
        # quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------
    # Define the prompt generation function using the local tokenizer
    # ----------------------------
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
    
    # ----------------------------
    # Apply LoRA configuration
    # ----------------------------
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # ----------------------------
    # Apply Prefix Tuning configuration
    # ----------------------------
    adapter_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection=True,
    )
    model = get_peft_model(model, adapter_config)
    
    # ----------------------------
    # Load and process datasets
    # ----------------------------
    train_dataset, val_dataset, _ = get_datasets(data_path, summaries_path)
    train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    val_dataset = val_dataset.map(generate_and_tokenize_prompt)
    
    # ----------------------------
    # Set up TrainingArguments (1 epoch for hyperparameter tuning)
    # ----------------------------
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,  # one epoch for quick evaluation
        learning_rate=learning_rate,
        fp16=True,
        save_total_limit=1,
        logging_steps=20,
        output_dir="output_dir",
        optim="paged_adamw_8bit",
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
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
    
    # ----------------------------
    # Run training and evaluate on validation set with error handling
    # ----------------------------
    try:
        trainer.train()
        eval_result = trainer.evaluate(eval_dataset=val_dataset)
        val_loss = eval_result["eval_loss"]
        print(f"Trial completed: val_loss = {val_loss}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Trial failed due to CUDA OOM, pruning trial.")
            torch.cuda.empty_cache()
            time.sleep(5)
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
    
    # Clean up GPU memory for next trial
    del model
    torch.cuda.empty_cache()
    time.sleep(5)
    
    return val_loss

# ----------------------------
# Run the Optuna study
# ----------------------------
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

with open("hyper-parameter_config_Llama-3.2-3B.json", "w") as f:
    json.dump(trial.params, f, indent=4)

print("Best parameters saved to hyper-parameter_config_Llama-3.2-3B.json")