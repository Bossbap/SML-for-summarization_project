import os

def process_files():
    baseline_dir = "baseline_mistral"
    cleaned_dir = "cleaned_lapresse_dataset"

    # Constant parts of the prompt.
    constant1 = "Résumes le texte suivant:\ntitre: "
    constant2 = "\n\ntexte: "

    errors = []  # List of tuples: (baseline_filename, reason)

    # Get .txt files from each folder and sort them for pairing.
    baseline_files = sorted(f for f in os.listdir(baseline_dir) if f.endswith('.txt'))
    cleaned_files = sorted(f for f in os.listdir(cleaned_dir) if f.endswith('.txt'))

    if len(baseline_files) != len(cleaned_files):
        print("Warning: The number of files in the two folders do not match.")

    for base_file, cleaned_file in zip(baseline_files, cleaned_files):
        base_path = os.path.join(baseline_dir, base_file)
        cleaned_path = os.path.join(cleaned_dir, cleaned_file)
        print(f"Processing baseline file: {base_file} with cleaned file: {cleaned_file}")

        # Compute title from cleaned file name (remove last 4 characters: ".txt")
        title = cleaned_file[:-4]

        # Read the cleaned file (for {text})
        try:
            with open(cleaned_path, "r", encoding="utf-8") as f:
                cleaned_text = f.read()
        except Exception as e:
            err_msg = f"Error reading cleaned file '{cleaned_file}': {e}"
            print(err_msg)
            errors.append((base_file, err_msg))
            # Clear baseline file and write "Aucun Résumé"
            try:
                with open(base_path, "w", encoding="utf-8") as f:
                    f.write("Aucun Résumé")
            except Exception as e2:
                print(f"Additionally, failed to clear baseline file '{base_file}': {e2}")
            continue

        # Calculate the expected prompt length:
        # prompt = constant1 + title + constant2 + cleaned_text
        prompt_length = len(constant1) + len(title) + len(constant2) + len(cleaned_text)

        # Read the baseline file.
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                baseline_content = f.read()
        except Exception as e:
            err_msg = f"Error reading baseline file '{base_file}': {e}"
            print(err_msg)
            errors.append((base_file, err_msg))
            # Clear baseline file and write "Aucun Résumé"
            try:
                with open(base_path, "w", encoding="utf-8") as f:
                    f.write("Aucun Résumé")
            except Exception as e2:
                print(f"Additionally, failed to clear baseline file '{base_file}': {e2}")
            continue

        # Check if the baseline file is long enough.
        if len(baseline_content) < prompt_length:
            err_msg = f"Baseline file '{base_file}' is shorter than the calculated prompt length."
            print(err_msg)
            errors.append((base_file, err_msg))
            try:
                with open(base_path, "w", encoding="utf-8") as f:
                    f.write("Aucun Résumé")
            except Exception as e2:
                print(f"Additionally, failed to clear baseline file '{base_file}': {e2}")
            continue

        # Extract the generated text (everything after the prompt).
        generated_text = baseline_content[prompt_length:]

        # Overwrite the baseline file with the generated text.
        try:
            with open(base_path, "w", encoding="utf-8") as f:
                f.write(generated_text)
            print(f"[OK] Processed file '{base_file}'.")
        except Exception as e:
            err_msg = f"Error writing to baseline file '{base_file}': {e}"
            print(err_msg)
            errors.append((base_file, err_msg))
            try:
                with open(base_path, "w", encoding="utf-8") as f:
                    f.write("Aucun Résumé")
            except Exception as e2:
                print(f"Additionally, failed to clear baseline file '{base_file}': {e2}")

    # Log all errors.
    if errors:
        print("\nFiles with issues:")
        for fname, reason in errors:
            print(f" - {fname}: {reason}")
    else:
        print("\nAll files processed successfully!")

    print(f"\nTotal number of files that didn't work: {len(errors)}")

if __name__ == "__main__":
    process_files()