from rouge_score import rouge_scorer

def score_summary_rouge(initial_text, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(initial_text, summary)
    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }


from bert_score import score
import torch
def score_summary_bertscore(initial_text, summary, lang="fr", device="cuda"):
    device = torch.device(device)
    P, R, F1 = score([summary], [initial_text], lang=lang, device=device)
    return F1.item()  # F1-score captures overall similarity


from transformers import pipeline

# grader = pipeline("text-classification", model="mistralai/Mistral-7B-Instruct-v0.3")

def score_summary_llm(initial_text, summary):
    prompt = f"""
    Évaluez la qualité du résumé suivant en lui attribuant une note de 0 à 10.
    
    Texte original:
    {initial_text}

    Résumé:
    {summary}

    Notez le résumé en fonction de sa fidélité, concision et clarté (0 = très mauvais, 10 = excellent).
    """
    score = grader(prompt)[0]["score"]  # Extract score
    return round(score * 10, 2)  # Convert to 0-10 scale

