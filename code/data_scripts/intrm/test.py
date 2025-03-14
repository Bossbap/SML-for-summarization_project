import os

# Function to check if a file has broken encoding
def check_encoding_validity(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read()
        return True  # No error → Encoding is valid
    except UnicodeDecodeError:
        return False  # Error → Encoding is broken

# Directory containing text files
input_dir = "data/cleaned_datasets/dataset_lapresse"  # Change this to your folder

# Track broken files
broken_files = []

# Iterate through all text files
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        if not check_encoding_validity(file_path):
            print(f"❌ Encoding error in: {filename}")
            broken_files.append(filename)

# Print summary
print(f"\n📊 Total broken files: {len(broken_files)}")