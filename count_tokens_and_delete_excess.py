import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

# Paths
DATASET_DIR = "data/cleaned_datasets/dataset_lapresse"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define token threshold
THRESHOLD = 2250  # Adjust as needed

# Load tokenizer (same as baseline)
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Same prompt format as in baseline script
PROMPT_TEMPLATE = "Résumes le texte suivant:\n\n{}"

# Get all .txt files in dataset directory
text_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(".txt")]
total_files = len(text_files)

# Store results
token_counts = []
deleted_files = 0

# Iterate through each file
for i, file_path in enumerate(text_files):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Format the prompt
        prompt = PROMPT_TEMPLATE.format(text)

        # Tokenize and count tokens
        num_tokens = len(tokenizer(prompt)["input_ids"])
        token_counts.append(num_tokens)

        if num_tokens > THRESHOLD:
            os.remove(file_path)  # Delete the file
            deleted_files += 1
            print(f"🗑 Deleted {file_path} | Tokens: {num_tokens} (Exceeded {THRESHOLD})")
        else:
            print(f"✅ Processed {i + 1}/{total_files}: {file_path} | Tokens: {num_tokens}")
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# Compute statistics if files remain
if token_counts:
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    avg_tokens = np.mean(token_counts)
    q1 = np.percentile(token_counts, 25)
    median = np.percentile(token_counts, 50)  # Q2
    q3 = np.percentile(token_counts, 75)

    print(f"📊 Token Statistics:\n"
          f" - Min: {min_tokens}\n"
          f" - Max: {max_tokens}\n"
          f" - Average: {avg_tokens:.2f}\n"
          f" - Q1 (25th percentile): {q1}\n"
          f" - Median (Q2): {median}\n"
          f" - Q3 (75th percentile): {q3}\n")

    # Define bins (starting at closest 250 below min_tokens)
    min_bin = (min_tokens // 250) * 250  # Round down to nearest 250
    bins = list(range(min_bin, max_tokens + 250, 250))  # Bin size of 250

    # Generate Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, label="Token Distribution")
    for rect in plt.gca().patches:
        height = rect.get_height()
        if height > 0:  # Only label non-empty bins
            plt.text(rect.get_x() + rect.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add vertical lines for average and quartiles
    plt.axvline(avg_tokens, color="red", linestyle="--", label=f"Mean: {avg_tokens:.1f}")
    plt.axvline(q1, color="green", linestyle="--", label=f"Q1: {q1:.1f}")
    plt.axvline(median, color="blue", linestyle="--", label=f"Median (Q2): {median:.1f}")
    plt.axvline(q3, color="purple", linestyle="--", label=f"Q3: {q3:.1f}")

    # Labels and Title
    plt.xlabel("Token Count Range")
    plt.ylabel("Number of Prompts")
    plt.title("Distribution of Token Counts in Prompts")
    plt.xticks(bins, rotation=45)  # Rotate x-axis labels for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()

    # Save the plot
    output_chart_path = os.path.join(OUTPUT_DIR, "token_count_histogram.png")
    plt.savefig(output_chart_path, dpi=300)
    plt.show()
    print(f"📊 Token count histogram complete. Chart saved as: {output_chart_path}")

print(f"🗑 Total files deleted: {deleted_files}")