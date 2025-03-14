import threading
from rouge_score import rouge_scorer
from bert_score import score
import os
import torch

def score_summary_rouge(initial_text, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(initial_text, summary)
    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }

torch.backends.cuda.enable_flash_sdp(False)
def score_summary_bertscore(initial_text, summary, lang="fr", device_type="cuda"):
    device = torch.device(device_type)
    with torch.amp.autocast(device_type=device_type):
        P, R, F1 = score([summary], [initial_text], lang=lang, device=device, nthreads=1)
    return F1.item()


def score_summary_llm(model, tokenizer, initial_text, summary):
    """
    Uses the Gemini API to evaluate the quality of a summary.
    
    Args:
        initial_text (str): The original text.
        summary (str): The summary to evaluate.
    
    Returns:
        float: Score between 0 and 1. (one is the best)
    """

    # Construct the prompt
    prompt = f"""
    Veuillez évaluer la qualité du résumé ci-dessous en lui attribuant une note finale comprise entre 0.00 et 10.00. Cette note doit refléter l’évaluation globale du résumé selon les critères suivants :

    Fidélité : Le résumé restitue-t-il correctement et complètement les informations essentielles du texte original ?
    Concision : Le résumé présente-t-il l'information de manière synthétique sans détails superflus ?
    Clarté : Le résumé est-il formulé de manière claire, compréhensible et structurée ?
    Contect : Le résumé est-il correctement contextualisé

    0.00 signifie « très mauvais »
    10.00 signifie « excellent »
    Répondez uniquement avec un nombre décimal à deux chiffres après la virgule (par exemple, 7.58).

    Texte original :
    {initial_text}

    ––––
    Résumé :
    {summary}
    ____

    Note (entre 0.00 et 10.00):
    """
    with tokenizer_lock:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)

    if len(output) == 0:
        print("No output for LLM score, initial_text or summary may have a problem.")
        return 0
    with tokenizer_lock:
        score_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].split()[0]
    
    try:
        score = float(score_text.strip())
    except ValueError:
        score = 0  # fallback if parsing fails

    if score > 10:
        print("Out of bound score")

    return round(score, 2) / 10
   
import torch.nn.functional as F
import multiprocessing as mp

manager = mp.Manager()
tokenizer_lock = manager.Lock()

def get_sentence_embedding(model, tokenizer, text, max_length=512):
    """
    Computes a sentence embedding by encoding the text and averaging the 
    last hidden states.
    """

    with tokenizer_lock:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        # Request hidden states; note: causal models may not have "pooler_output"
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    # Use the last hidden layer (shape: [batch_size, seq_len, hidden_dim])
    last_hidden = outputs.hidden_states[-1]
    # Average across the sequence length to get a fixed-size embedding
    embedding = last_hidden.mean(dim=1)
    return embedding

def score_summary_ruse(model, tokenizer, original_text, summary_text):
    """
    Computes a RUSE-like score using a causal LM from Hugging Face:
      1. Get embeddings for both texts by averaging their final hidden states.
      2. Compute the cosine similarity.
      3. Transform the cosine similarity from [-1, 1] to a 0–1 scale.

      return score (0, 1), embedding diff
    """
    emb_orig = get_sentence_embedding(model, tokenizer, original_text)
    emb_sum = get_sentence_embedding(model, tokenizer, summary_text)
    
    # Compute cosine similarity between embeddings (returns a tensor of shape [1])
    cosine_sim = F.cosine_similarity(emb_orig, emb_sum)
    # Transform similarity from [-1, 1] to [0, 1]
    score = ((cosine_sim + 1) / 2)
    return round(score.item(), 2), emb_orig - emb_sum
