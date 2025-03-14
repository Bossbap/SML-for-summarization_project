import os

# Function to check if a file has broken encoding
def check_encoding_validity(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read()
        return True  # Encoding is valid
    except UnicodeDecodeError:
        return False  # Encoding is broken

input_dir = "data/cleaned_datasets/dataset_lapresse"
broken_files = []

# Iterate through all text files
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        if not check_encoding_validity(file_path):
            print(f"Encoding error in: {filename}")
            broken_files.append(filename)

# Print summary
print(f"\nTotal broken files: {len(broken_files)}")