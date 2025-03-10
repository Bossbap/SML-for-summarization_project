import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, PrefixTuningConfig, get_peft_model, TaskType

# Define the objective function for Bayesian Optimization
def objective(trial):
    # 🔍 Define hyperparameter search space
    num_virtual_tokens = trial.suggest_int("num_virtual_tokens", 10, 100, step=10)
    lora_alpha = trial.suggest_int("lora_alpha", 16, 64, step=16)
    lora_dropout = trial.suggest_float("lora_dropout", 0.01, 0.2)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    warmup_steps = trial.suggest_int("warmup_steps", 10, 200, step=10)

    # ✅ Prefix Tuning Config
    prefix_config = PrefixTuningConfig(
        peft_type="PREFIX_TUNING",
        task_type="CAUSAL_LM",
        num_virtual_tokens=num_virtual_tokens,
        token_dim=4096,  # Must match LLaMA 3B
        num_transformer_submodules=1,
        num_attention_heads=32,
        num_layers=26,
    )

    # ✅ LoRA Config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Keeping fixed
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # ✅ Apply LoRA & Prefix Tuning
    model = get_peft_model(model, lora_config)
    model = get_peft_model(model, prefix_config)

    # ✅ Training Arguments
    training_args = TrainingArguments(
        output_dir=f"data/models/hyperopt_trial_{trial.number}",
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,  # Keep fixed for fair comparison
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        fp16=True,
        push_to_hub=False
    )

    # ✅ Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # ✅ Train & Evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    return eval_results["eval_loss"]  # Minimize loss

study = optuna.create_study(direction="minimize")  # Minimize validation loss
study.optimize(objective, n_trials=30)

# ✅ Print Best Hyperparameters
print("Best Hyperparameters:", study.best_params)
