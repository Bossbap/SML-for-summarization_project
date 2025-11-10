# Fine-Tuning Mistral with Curriculum Learning  
Baptiste Geisenberger & Enzo Pinchon

github link: https://github.com/Bossbap/SML-for-summarization_project

## Overview  
This project fine-tunes the **Mistral** model using **LoRA** and **Curriculum Learning** to improve text summarization quality. The dataset is processed dynamically, ranking examples from easiest to hardest.  

## Project Structure  
├── code/ # Contains all scripts for training, evaluation, and utilities
└── data/ # Dataset folder (must be downloaded separately)

## Setup  
1. **Clone the repository & install dependencies:**  
   ```bash
   git clone <repo_url>  
   cd <repo_name>  
   pip install -r requirements.txt

## Download the dataset:
Donwload our curated dataset from [Drive Link](https://drive.google.com/drive/u/0/folders/1RVb7e_dyLVbgpN5AF7MstELhr9mbmGWj) and place it in your project root to run the code.