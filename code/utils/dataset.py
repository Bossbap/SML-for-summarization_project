import os
import glob
import tqdm
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def load_data(data_dir, summaries_path, **args) -> list[tuple]:
    """
    Load text data and their corresponding summaries from a specified directory,
    generating summaries when necessary.

    This function processes all `.txt` files in the given `data_dir`. For each file,
    it attempts to load an associated summary from the directory specified by `summaries_path`.
    If the summary file does not exist or if forced regeneration is enabled, it will generate
    a summary using the provided summarization function and save it.

    Parameters:
        data_dir (str): Path to the directory containing text files (.txt) to process.
        summaries_path (str): Path to the directory where summary files are stored.

        **args: Additional keyword arguments.
            - LOAD_SUMMARIZED_TEXT (bool, optional): 
                  Flag indicating whether to load existing summary files if they exist.
                  Default is True.
            
            - FORCE_SUMMARY_REGENERATION (bool, optional): 
                  If True, forces regeneration of the summary even if a summary file exists.
                  Default is False.

            - summariation_func (callable, optional): 
                  A function that takes a text string as input and returns a summary.
                  Default is a lambda that returns an Exception ("No summarization function").

    Returns:
        list[tuple]: A list of tuples, each containing:
            - text (str): The original text content from the file.
            - summary (str): The corresponding summary (either loaded or generated).

    Behavior:
        1. Retrieves all `.txt` file paths from `data_dir` using glob.
        2. For each file:
            - Reads the full text content.
            - Determines the corresponding summary file path in `summaries_path`.
            - If `LOAD_SUMMARIZED_TEXT` is True and `FORCE_SUMMARY_REGENERATION` is False:
                  a. If the summary file exists, it loads the summary.
                  b. If not, it generates the summary using `summariation_func`,
                     writes it to the summary file, and then loads it.
            - If `LOAD_SUMMARIZED_TEXT` is False or `FORCE_SUMMARY_REGENERATION` is True:
                  a. If a summary file exists and regeneration is not forced, it loads the summary.
                  b. Otherwise, it generates the summary, writes it to the file.
        3. Appends a tuple `(text, summary)` for each processed file to the result list.
        4. Returns the list of `(text, summary)` tuples.

    Raises:
        Exception: If the `summariation_func` returns an Exception when attempting to generate a summary.

    Dependencies:
        - os, glob: For file and path operations.
        - tqdm: For displaying a progress bar during file processing.
    """

    LOAD_SUMMARIZED_TEXT = args.get('LOAD_SUMMARIZED_TEXT', True)
    FORCE_SUMMARY_REGENERATION = args.get('FORCE_SUMMARY_REGENERATION', False)
    summarization = args.get("summariation_func", lambda x: Exception("No summarization function"))

    data = []
    file_paths = glob.glob(os.path.join(data_dir, "*.txt"))
    for file in tqdm.tqdm(file_paths):
        with open(file, encoding="utf-8") as f:
            text = f.read()

        if LOAD_SUMMARIZED_TEXT and not FORCE_SUMMARY_REGENERATION:
            summary_file = os.path.join(summaries_path, os.path.basename(file))

            if not os.path.exists(summary_file):
                print(f"[summary not found] Summarizing {file}...")
                summary = summarization(text)
                if isinstance(summary, Exception): raise summary
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(summary)

            with open(summary_file, "r", encoding="utf-8") as f:
                summary = f.read()
        else:
            summary_file = os.path.join(summaries_path, os.path.basename(file))

            if not FORCE_SUMMARY_REGENERATION and os.path.exists(summary_file):
                print("[summary already exist] A summary already, auto-loading it. Set FORCE_SUMMARY_REGENERATION=False to regenerate it.")
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = f.read()

            else:
                summary = summarization(text)
                if isinstance(summary, Exception): raise summary

                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(summary)

        data.append((text, summary))
    return data

class SummarizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        orig, summ = self.data[idx]
        return orig, summ

def get_datasets(dataset_path, summaries_path, random_seed=1):
    data = load_data(dataset_path, summaries_path)

    # stratify the data based on unsummarized text length
    lengths = np.array([len(file[0]) for file in data])
    bins = np.histogram_bin_edges(lengths, bins='auto')
    num_bins_initial = len(bins)
    binned_lengths = np.digitize(lengths, bins)

    # ensure no bin has more than 2 samples
    unique, counts = np.unique(binned_lengths, return_counts=True)
    valid_bins = {u for u, c in zip(unique, counts) if c >= 2}

    # merge rare bins to avoid trying to split on a too small bin
    for i in range(len(binned_lengths)):
        if binned_lengths[i] not in valid_bins:
            binned_lengths[i] = min(valid_bins, key=lambda x: abs(x - binned_lengths[i]))
            
    # recalculate the bins (using valid_bins)
    num_bins = len(valid_bins)
    bins = np.array([bin_start for idx, bin_start in enumerate(bins) if (idx+1) in valid_bins], dtype=np.float32)

    lengths = np.array([len(file[0]) for file in data])
    quantiles = np.percentile(lengths, [33, 66])
    coarse_strata = np.digitize(lengths, bins=quantiles)

    train_data, temp_data, train_strata, temp_strata = train_test_split(
        data, coarse_strata, test_size=0.4, random_state=random_seed, stratify=coarse_strata
    )

    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_strata, test_size=0.5, random_state=random_seed, stratify=temp_strata
    )

    train_dataset = SummarizationDataset(train_data)
    val_dataset   = SummarizationDataset(val_data)
    test_dataset  = SummarizationDataset(test_data)

    return train_dataset, val_dataset, test_dataset
