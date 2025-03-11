import os
import glob
import tqdm
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def load_data(data_dir, summaries_path, **args) -> list[tuple]:
    """
    Load text data and their corresponding summaries from a specified directory,
    generating summaries when necessary. Includes the filename along with the text and summary.

    Parameters and behavior are similar to the previous version, but now the function
    also returns the filename for each data entry.

    Returns:
        list[tuple]: A list of tuples, each containing:
            - filename (str): The name of the text file.
            - text (str): The original text content from the file.
            - summary (str): The corresponding summary (either loaded or generated).
    """
    LOAD_SUMMARIZED_TEXT = args.get('LOAD_SUMMARIZED_TEXT', True)
    FORCE_SUMMARY_REGENERATION = args.get('FORCE_SUMMARY_REGENERATION', False)
    summarization = args.get("summariation_func", lambda x: Exception("No summarization function"))

    data = []
    file_paths = glob.glob(os.path.join(data_dir, "*.txt"))
    
    for file in tqdm.tqdm(file_paths):
        with open(file, encoding="utf-8") as f:
            text = f.read()

        # Retrieve the filename (basename) for this file
        filename = os.path.basename(file)

        if LOAD_SUMMARIZED_TEXT and not FORCE_SUMMARY_REGENERATION:
            summary_file = os.path.join(summaries_path, filename)

            if not os.path.exists(summary_file):
                print(f"[summary not found] Summarizing {filename}...")
                summary = summarization(text)
                if isinstance(summary, Exception): raise summary
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(summary)

            with open(summary_file, "r", encoding="utf-8") as f:
                summary = f.read()
        else:
            summary_file = os.path.join(summaries_path, filename)

            if not FORCE_SUMMARY_REGENERATION and os.path.exists(summary_file):
                print(f"[summary already exist] A summary already, auto-loading it. Set FORCE_SUMMARY_REGENERATION=False to regenerate it.")
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = f.read()

            else:
                summary = summarization(text)
                if isinstance(summary, Exception): raise summary

                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(summary)

        # Append a tuple (filename, text, summary)
        data.append((filename, text, summary))
    return data
import warnings

class SummarizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.map_func = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename, orig, summ = self.data[idx]
        
        if self.map_func:
            return self.map_func(filename, orig, summ)
        return filename, orig, summ
    
    def map(self, map_func):
        if not callable(map_func):
            raise Exception("map_function should be callable, current type:", type(map_func))
        
        if self.map_func is not None:
            warnings.warn("A map_func is already defined, it will be overwrite", stacklevel=2)

        self.map_func = map_func

def get_datasets(dataset_path, summaries_path, random_seed=1):
    data = load_data(dataset_path, summaries_path)

    # stratify the data based on unsummarized text length
    lengths = np.array([len(file[1]) for file in data])
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

    lengths = np.array([len(file[1]) for file in data])
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
