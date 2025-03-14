import os
import difflib
import re

def simple_normalize(text):
    """
    Normalize text by replacing any whitespace (newline, tab, multiple spaces)
    with a single space and stripping leading/trailing whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()

def process_files():
    baseline_dir = "baseline"
    cleaned_dir = "cleaned_lapresse_dataset"
    prompt = "Résumes le texte suivant:\n\n"

    # List .txt files and sort to ensure consistent order.
    baseline_files = sorted(f for f in os.listdir(baseline_dir) if f.endswith('.txt'))
    cleaned_files = sorted(f for f in os.listdir(cleaned_dir) if f.endswith('.txt'))

    if len(baseline_files) != len(cleaned_files):
        print("Warning: The number of files in the two folders do not match.")

    for base_file, cleaned_file in zip(baseline_files, cleaned_files):
        base_path = os.path.join(baseline_dir, base_file)
        cleaned_path = os.path.join(cleaned_dir, cleaned_file)
        print(f"Processing baseline file: {base_file} with cleaned file: {cleaned_file}")

        with open(base_path, "r", encoding="utf-8") as f:
            base_content = f.read()
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_content = f.read()

        # Remove the fixed prompt if present.
        if base_content.startswith(prompt):
            baseline_remaining = base_content[len(prompt):]
        else:
            print(f"Warning: {base_file} does not start with the expected prompt.")
            baseline_remaining = base_content

        # Normalize both texts.
        norm_baseline = simple_normalize(baseline_remaining)
        norm_cleaned = simple_normalize(cleaned_content)

        # Check if the start of norm_baseline is similar to norm_cleaned.
        prefix_of_baseline = norm_baseline[:len(norm_cleaned)]
        matcher = difflib.SequenceMatcher(None, norm_cleaned, prefix_of_baseline)
        similarity = matcher.ratio()

        # Decide where the cleaned text ends in the original baseline_remaining.
        if similarity >= 0.9:
            # Assume the cleaned text is at the beginning.
            # We use the length of the cleaned file (original form) as an approximation.
            offset = len(cleaned_content)
        else:
            # Try to locate a snippet from the cleaned text in the baseline_remaining.
            snippet = norm_cleaned[:50]  # first 50 characters
            snippet_index = baseline_remaining.find(snippet)
            if snippet_index == -1:
                print(f"Warning: Could not locate a snippet from the cleaned text in {base_file}.")
                offset = 0  # fallback: keep entire file
            else:
                offset = snippet_index + len(snippet)

        # Extract the generated text as everything after the cleaned text portion.
        generated_text = baseline_remaining[offset:]

        # Write back only the generated text.
        with open(base_path, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"Finished processing {base_file}.\n")

if __name__ == "__main__":
    process_files()