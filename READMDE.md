# SML for French Text Summarization

## Project Overview
This project focuses on **fine-tuning a Small Language Model (SML)** for the task of **French text summarization**, using a dataset of geopolitical articles. The goal is to train a lightweight model capable of generating accurate and concise summaries, making it suitable for resource-constrained environments.

## Project Workflow

### 1. Data Collection
- A set of approximately **5,000 articles** was collected from **Wikipedia** using a custom crawling process.
- These articles focus on **geopolitics**, ensuring a domain-specific training corpus.

### 2. Data Preprocessing
- Articles are cleaned to remove noise (e.g., boilerplate, citations, irrelevant sections).
- Language filtering ensures all articles are in **French**.
- Basic statistics (length, word count) are computed to understand the dataset.
- Only the **newest version** of each article is retained if multiple versions exist.
- The **top 5000 longest articles** are selected to maximize summarization challenge.

### 3. Summary Generation
- For each article, a summary is generated using a **regular-sized LLM** (e.g., Mistral, LLaMA, or similar). 
- These generated summaries form the **training labels** for the fine-tuning process.

### 4. Fine-Tuning
- A **Small Language Model (SML)** (e.g., distilBART, mBART, or similar lightweight model) is fine-tuned on the `article → summary` pairs.
- The goal is to train a model that is **faster and cheaper** than larger LLMs, while maintaining good performance.

### 5. Evaluation
- The performance of the fine-tuned SML is evaluated using:
    - **ROUGE scores** (to compare generated vs reference summaries)
    - **LLM-as-a-judge** (a larger model rates the quality of the summaries)
- The fine-tuned SML is compared to a **baseline model** (out-of-the-box summarization model or zero-shot approach).

---

## Data Source
- Articles: Crawled from **Wikipedia**, filtered for geopolitical relevance.
- Original dataset stored in: `../dataset/original_dataset` (5k `.txt` files).
- Processed dataset stored in: `../dataset/cleaned_articles.csv`.

---

##  Tools & Technologies

## Folder Structure