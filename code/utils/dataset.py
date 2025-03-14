import os
import glob
import unicodedata
import tqdm
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def load_data(data_dir, summaries_path, strict=True, filename_parsing=None, **args) -> list[tuple]:
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
    if len(file_paths) == 0:
        raise Exception("No file founds at: " + data_dir + "\nCurrent path: " + os.getcwd())
    for file in tqdm.tqdm(file_paths):
        with open(file, encoding="utf-8") as f:
            text = f.read()

        # Retrieve the filename (basename) for this file
        filename = os.path.basename(file)
        filename = unicodedata.normalize('NFC', filename)

        if callable(filename_parsing):
            filename = filename_parsing(filename).strip()

        if LOAD_SUMMARIZED_TEXT and not FORCE_SUMMARY_REGENERATION:
            summary_file = os.path.join(summaries_path, filename)
            if not os.path.exists(summary_file):

                summary = summarization(text)
                if isinstance(summary, Exception): 
                    if strict: raise summary
                    continue
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
                if strict:
                    raise Exception("Summary not found: ", filename, "strict mode is enable and LOAD_SUMMARIZED_TEXT is false (for auto generation)")
                continue

        # Append a tuple (filename, text, summary)
        data.append((filename, text, summary))
    return data
import warnings

class SummarizationDataset(Dataset):
    """
        Manage dataset

        Can manage pair dataset with rank, however this option is not robust (really static):
        - first dataset score will every time take the first value of data/label_summaries_scores.csv
        - sd dataset will take the sd one

        data/label_summaries_scores.csv should be: filename | score_1 | score_2
    """
    def __init__(self, data, sd_data=None):
        self.is_pair_dataset = sd_data is not None
        self.keys = None
        self.scores = {}

        if self.is_pair_dataset:
            self.data = {row[0]: [row] for row in data}
            for row in sd_data:
                if row[0] in self.data: self.data[row[0]].append(row)
            self.keys = list(self.data.keys())

            # get score for ranking pairs

        else: self.data = data
        self.map_func = None
        self.output_format = tuple
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_pair_dataset:
            filename = self.keys[idx]
            data_point_1, data_point_2 = self.data[filename]

            scores = self.scores.get(filename, {})
            if scores.get("score_1") > scores.get("score_2"):
                _, orig, best_summ = data_point_1
                _, _, sd_summ = data_point_2
                best_score = scores.get("score_1")
                second_score = scores.get("score_2")
            else:
                _, orig, best_summ = data_point_2
                _, _, sd_summ = data_point_1
                best_score = scores.get("score_2")
                second_score = scores.get("score_1")
            
        else: filename, orig, summ = self.data[idx]

        if self.output_format is tuple:
            output = filename, orig, summ
        elif self.output_format is list:
            output = [filename, orig, summ]
        elif self.output_format is dict:
            if self.is_pair_dataset:
                output = {"filename": filename, "initial_text": orig, "summary_best": best_summ, "second_summary": sd_summ, "score_best": best_score, "second_score": second_score}
            else: output = {"filename": filename, "initial_text": orig, "summary": summ}
        else:
            raise Exception("Unknow output type")
        
        if self.map_func:
            return self.map_func(output)
        return output
    
    def curriculum_setup(self, csv_path):
        if not self.is_pair_dataset: raise Exception("Sort work only for paired dataset")
        df = pd.read_csv(csv_path)
        difficulty_map = dict(zip(df["filename"], df["avg_difficulty"]))
        self.keys.sort(key=lambda k: difficulty_map.get(k, float("inf")))
        for _, row in df.iterrows():
            self.scores[row["filename"]] = {"score_1": row["score_1"], "score_2": row["score_2"]}
        return self

    def map(self, map_func):
        if not callable(map_func):
            raise Exception("map_function should be callable, current type:", type(map_func))
        
        if self.map_func is not None:
            warnings.warn("A map_func is already defined, it will be overwrite", stacklevel=2)

        self.map_func = map_func
        return self

    def formatted_as(self, output_type: type):
        assert output_type in [dict, tuple, list], "Impossible to format with type: " + output_type.__name__

        self.output_format = output_type

        return self

def get_datasets(dataset_path, summaries_path, random_seed=1, strict=True):
    sd_summaries_path = None
    if type(summaries_path) is tuple and len(summaries_path) == 2:
        summaries_path, sd_summaries_path = summaries_path

    data = load_data(dataset_path, summaries_path, strict=strict)

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
        data, coarse_strata, test_size=0.2, random_state=random_seed, stratify=coarse_strata
    )

    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_strata, test_size=0.5, random_state=random_seed, stratify=temp_strata
    )


    if sd_summaries_path is None: sd_data = None
    else: sd_data = load_data(dataset_path, sd_summaries_path, strict=strict)

    train_dataset = SummarizationDataset(train_data, sd_data)
    val_dataset   = SummarizationDataset(val_data, sd_data)
    test_dataset  = SummarizationDataset(test_data, sd_data)

    return train_dataset, val_dataset, test_dataset