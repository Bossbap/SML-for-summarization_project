# Fine-Tuning Small Language Models for French Summarization
Enzo Pinchon · Baptiste Geisenberger  

## Why This Project?
We tackle automatic text summarization for French news under tight compute constraints. Instead of relying on heavy general-purpose LLMs, we curate ~5k La Presse articles, generate high-quality target summaries with Mistral 24B & Qwen2.5 32B, and fine-tune deployable Small Language Models (≤7B parameters). We experiment with LoRA, Prefix-Tuning, and a curriculum setup driven by ROUGE/RUSE-based difficulty. The best LoRA + curriculum variant of LLaMA‑3.2‑3B matches the teacher LLMs on ROUGE/BERTScore, while staying within a single RTX A4000 (16 GB) using 4–8 bit quantization via BitsAndBytes.

## Repository Map
```
code/
  data_scripts/        # scraping, cleaning, token filtering utilities
  data_process/        # notebooks to build the final train/val/test splits
  finetune/            # LoRA, Prefix, curriculum training + inference scripts
  utils/               # dataset loader, scoring helpers
baseline_*.py          # zero-shot baselines for Mistral & LLaMA
requirements.txt
Report.pdf             # long-form write-up with full plots/details
```

## Prerequisites & Setup
1. **Python & CUDA** – Python 3.10+ and CUDA 12.x were used. Training expects a single NVIDIA GPU with ≥16 GB VRAM; CPU-only mode is only feasible for preprocessing/evaluation.
2. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Hugging Face access** – Authenticate to download gated checkpoints:
   ```bash
   huggingface-cli login
   ```
4. **Folder expectations** – All scripts assume paths relative to the repo root (see `# RUN FROM /` comments). Keep the following structure:
   ```
   data/
     raw_lapresse_dataset/                # raw .txt articles
     cleaned_datasets/dataset_lapresse/   # cleaned inputs produced by preprocessing.py
     summaries_datasets/mistral-summaries/# gold summaries (teacher LLM)
     models/                              # checkpoints saved by fine-tuning scripts
   ```

## Data Sources
- **Ready-to-use dataset** – Download our cleaned corpus + Mistral summaries from [Google Drive](https://drive.google.com/drive/u/0/folders/1RVb7e_dyLVbgpN5AF7MstELhr9mbmGWj). Unzip so that articles land in `data/cleaned_datasets/dataset_lapresse` and summaries in `data/summaries_datasets/mistral-summaries`.
- **Optional raw rebuild**
  1. Scrape geopolitics-oriented Wikipedia pages with `code/data_scripts/wikipedia_scraper.py`.
  2. Inspect/deduplicate La Presse with `code/data_scripts/lapresse_inspect.ipynb`.
  3. Count tokens and drop over-long files with `code/data_scripts/count_tokens_and_delete_excess.py`.

## Running the Pipeline
1. **Preprocess raw text**  
   Cleans HTML, enforces French-only content, and trims length.
   ```bash
   python code/data_scripts/preprocessing.py
   ```
   Outputs land in `data/cleaned_datasets/dataset_lapresse/`.

2. **(Re)generate silver summaries (if you skipped the Drive download)**  
   - Use `code/baseline_mistral.py` or `code/baseline_llama.py` for quick zero-shot baselines.  
   - For high-quality targets, run the notebook `code/data_process/generating_dataset.ipynb` and plug in Mistral 24B / Qwen2.5 32B via an HF Inference Endpoint or a larger GPU box. Save summaries under `data/summaries_datasets/<teacher-name>/`.

3. **Create train/val/test splits**  
   `code/data_process/process_dataset.ipynb` stratifies by article length to keep distributions aligned across splits.

4. **Fine-tune the SLMs (run from repo root)**  
   - LoRA on LLaMA‑3.2‑3B:
     ```bash
     python code/finetune/LORA-fine-tune_Llama-3.2-3B.py
     ```
   - LoRA on Mistral‑7B:
     ```bash
     python code/finetune/LORA-fine-tune_Mistral-7B.py
     ```
   - Prefix-tuning equivalents live in the same folder (`prefix-fine-tune_*.py`).  
   Tweak paths at the top of each script to switch datasets, summaries, or save locations. Training arguments were tuned with the provided Optuna HPO scripts (`HPO-*.py`).

5. **Paired curriculum & pairwise loss (optional)**  
   Use `code/finetune/paired_curriculum.ipynb` to build difficulty scores (based on ROUGE, RUSE, LLM-as-judge) and feed them into `SummarizationDataset.curriculum_setup`.

6. **Inference with fine-tuned adapters**
   ```bash
   python code/finetune/inference.py               # LoRA adapter + base model
   python code/finetune/inference_from_checkpoint.py  # for fully saved checkpoints
   ```
   Generated summaries are written to `data/generated_summaries_fine-tuned_models/<run-name>/`.

7. **Evaluation**
   - `code/utils/scores.py` exposes helpers for ROUGE, BERTScore, LLM-as-judge (Mistral 7B prompt) and the custom RUSE implementation.
   - For quick checks, import the functions in a notebook or script:
     ```python
     from code.utils.scores import score_summary_rouge, score_summary_ruse
     ```
   - Full analysis plots (distribution, PCA, correlations) are documented in `Report.pdf`.

## Key Findings
- LoRA on LLaMA‑3.2‑3B reaches ROUGE‑1/2/L ≈ 0.44 and BERTScore ≈ 0.90 when trained with the curriculum pairs, matching the 24B teacher despite a 10× memory reduction.
- Prefix-tuned Mistral‑7B nearly closes the gap without modifying base weights, hinting at strong inherent summarization ability.
- Traditional lexical metrics reward long outputs; the custom RUSE + LLM-as-judge combo better reflects factual faithfulness and conciseness.

## Troubleshooting
- **CUDA OOM** – Lower `per_device_train_batch_size`, shorten `max_length` in `generate_and_tokenize_prompt`, or tweak BitsAndBytes precision.
- **Unicode / encoding errors** – File names must be NFC-normalized UTF‑8; see `load_data` in `code/utils/dataset.py`.
- **Slow evaluations** – Disable FlashAttention for metric runs (already handled in `scores.py`) and run ROUGE on CPU if GPU memory is saturated.

For experiment logs, hyper-parameter sweeps, and plots, consult `Report.pdf`. The README intentionally stays concise; the report dives into methodology, ablations, and future work (adapters, RLHF, larger curriculum sets).
