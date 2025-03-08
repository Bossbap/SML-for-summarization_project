import os
import re
import langdetect
from bs4 import BeautifulSoup

# Function to clean text
def clean_text(text, stats, max_number_char):
    # Check if text is in French
    try:
        if langdetect.detect(text) != "fr":
            stats["non_french"] += 1
            return None  # Drop file if not French
    except langdetect.LangDetectException:
        stats["non_french"] += 1
        return None  # Drop file if language detection fails

    # Drop articles above 20000 characters
    if len(text) > max_number_char:
        stats["too_long"] += 1
        return None

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra spaces, tabs, blank lines, unknown symbols (keep punctuation)
    text = re.sub(r"[^\w\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ.,;:!?'\"()-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces

    # Normalize line breaks
    text = re.sub(r"(\n\s*)+", "\n", text)

    # Remove everything if "Notes et références" appears alone on a line
    text = re.sub(r"\n+Notes et références\s*\n.*", "", text, flags=re.DOTALL)

    return text.strip()

def preprocess(input_dir, output_dir, max_number_char):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Statistics for dropped files
    stats = {"too_long": 0, "non_french": 0, "total_files": 0, "processed_files": 0}

    # Process all .txt files
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            stats["total_files"] += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read file
            with open(input_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Clean the text
            cleaned_text = clean_text(content, stats, max_number_char)

            # Save if valid (not dropped)
            if cleaned_text:
                stats["processed_files"] += 1
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(cleaned_text)

    # Print statistics
    print(f"\n📊 Preprocessing complete for {input_dir}:")
    print(f"   📝 Total articles processed: {stats['total_files']}")
    print(f"   ✅ Successfully cleaned & saved: {stats['processed_files']}")
    print(f"   ⚠️ Dropped (too long): {stats['too_long']}")
    print(f"   ⚠️ Dropped (not in French): {stats['non_french']}\n")

# Directories
input_dir_wikipedia = "data/datasets/dataset_wikipedia"
output_dir_wikipedia = "data/cleaned_datasets/dataset_wikipedia"

input_dir_lapresse = "data/datasets/dataset_lapresse"
output_dir_lapresse = "data/cleaned_datasets/dataset_lapresse"

preprocess(input_dir_wikipedia, output_dir_wikipedia, 30000)
preprocess(input_dir_lapresse, output_dir_lapresse, 20000)
