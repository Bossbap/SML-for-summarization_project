import os
import unicodedata

def clean_filename(filename):
    """Normalize filenames and remove corrupted characters."""
    normalized_name = unicodedata.normalize("NFC", filename)
    cleaned_name = normalized_name.encode("ascii", "ignore").decode("utf-8")  # Remove non-ASCII characters
    return cleaned_name

def rename_files(directory):
    for filename in os.listdir(directory):
        new_filename = clean_filename(filename)
        if filename != new_filename:
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")

# Apply renaming to both article and summary folders
rename_files("results/baseline/summaries_lapresse")

print("✅ Filenames cleaned and normalized.")