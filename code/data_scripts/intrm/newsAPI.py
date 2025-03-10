import requests
import json
import os
from bs4 import BeautifulSoup

# Remplace 'YOUR_API_KEY' par ta clé API NewsAPI
API_KEY = '6cf8191dec404d1ba6c1f48a3ae3f27e'
BASE_URL = 'https://newsapi.org/v2/everything'

# Fonction pour récupérer le contenu complet de l'article à partir de l'URL
def get_full_article(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Ici, tu peux ajuster la manière dont tu extraits le texte complet en fonction du site
        # Exemple avec Le Monde : on cherche le contenu de l'article dans une balise <div> spécifique
        article_content = soup.find('div', {'class': 'article-body'})  # Change si nécessaire
        
        if article_content:
            return article_content.get_text(strip=True)
        else:
            print(f"Le contenu de l'article à {url} n'a pas pu être extrait.")
            return None
    else:
        print(f"Erreur de récupération de la page : {response.status_code}")
        return None

# Fonction pour obtenir des articles sur la géopolitique en français
def get_geopolitics_articles(page=1):
    params = {
        'q': 'géopolitique',  # Rechercher "géopolitique" dans les articles
        'language': 'fr',  # Limiter aux articles en français
        'pageSize': 100,  # Nombre maximum d'articles par requête
        'page': page,  # Numéro de la page pour pagination
        'apiKey': API_KEY,  # Clé API
    }

    # Faire la requête à NewsAPI
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        print(f"Erreur lors de l'appel API: {response.status_code}")
        return []

# Fonction pour filtrer les articles par longueur (plus de 300 mots)
def is_article_long_enough(article_content):
    word_count = len(article_content.split())
    return word_count >= 300

# Fonction pour sauvegarder les articles dans des fichiers .json
def save_articles_to_folder(articles, folder_name="dataset2"):
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for i, article in enumerate(articles, 1):
        # Récupérer le contenu complet de l'article à partir de l'URL
        full_content = get_full_article(article['url'])
        
        if full_content and is_article_long_enough(full_content):
            # Ajouter le contenu complet à l'article
            article['full_content'] = full_content

            # Définir le nom du fichier (par exemple article_1.json)
            file_name = os.path.join(folder_name, f"article_{i}.json")
            
            # Sauvegarder l'article en format JSON
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(article, f, ensure_ascii=False, indent=4)
            print(f"Article {i} sauvegardé sous {file_name}")
        else:
            print(f"Article {i} n'a pas pu être sauvegardé (contenu incomplet ou trop court).")

# Fonction pour afficher les articles filtrés
def display_articles(articles):
    for i, article in enumerate(articles, 1):
        print(f"Article {i}:")
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']['name']}")
        print(f"Published At: {article['publishedAt']}")
        print(f"URL: {article['url']}")
        print(f"Description: {article['description']}")
        
        # Afficher le contenu complet si disponible
        if 'full_content' in article:
            print(f"Full Content:\n{article['full_content'][:500]}...")  # Affichage partiel du contenu pour la lisibilité
        print("-" * 100)

# Main
if __name__ == "__main__":
    total_articles_saved = 0
    page = 1
    while total_articles_saved < 5000:
        # Étape 1 : Récupérer les articles sur la géopolitique
        articles = get_geopolitics_articles(page)
        
        if articles:
            # Étape 2 : Sauvegarder les articles dans des fichiers .json si leur contenu est suffisamment long
            save_articles_to_folder(articles)
            
            # Compter combien d'articles ont été sauvegardés dans cette page
            total_articles_saved += len(articles)
            print(f"Total d'articles sauvegardés: {total_articles_saved}")
            
            # Passer à la page suivante pour continuer à récupérer plus d'articles
            page += 1
        else:
            print("Aucun article trouvé, arrêt du processus.")
            break