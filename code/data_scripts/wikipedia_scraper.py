import os
import re
import wikipediaapi
import ollama
import concurrent.futures
import threading
from queue import Queue

# Constants
LANGUAGE = "fr"
START_PAGE = "Relations_internationales"
DATASET_DIR = "dataset2"
MAX_PAGES = 6000
MIN_CHAR_COUNT = 1500

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(user_agent='Baptiste', language=LANGUAGE)

# Create dataset directory if it doesn't exist
os.makedirs(DATASET_DIR, exist_ok=True)

# Thread-safe queue for URLs to visit
queue = Queue()
queue.put(START_PAGE)
visited = set()
saved_count = 0
lock = threading.Lock()

# Function to classify text using Ollama asynchronously
def classify_text(text):
    if len(text) < MIN_CHAR_COUNT:
        print(f"Rejected (too short - {len(text)} characters): {text[:100]}...")
        return False
    
    prompt = f"""Tu es un expert en classification de texte. Ton rôle est de déterminer si le texte suivant est directement lié à la géopolitique. Par exemple, un texte parlant d'événements historiques internationaux importants, de concepts politiques ou géographiques, ou d'une personne politique importante sont tous des textes directement liés à la géopolitique. Voici le texte:

{text}

Ce texte est-il directement en lien avec la géopolitique?
Formule ton raisonnement en quelques lignes, puis répond "VRAI" ou "FAUX" à la dernière ligne en fonction de ta réponse."""
    
    response = ollama.chat(model="mistral:latest", messages=[{"role": "user", "content": prompt}])
    response_text = response['message']['content']
    print(f"Ollama response: {response_text}")  # Debug output
    match = re.search(r'VRAI|FAUX', response_text, re.IGNORECASE)
    return match and match.group(0).upper() == "VRAI"

# Function to process a Wikipedia page
def process_page():
    global saved_count
    while saved_count < MAX_PAGES:
        page_title = queue.get()
        if page_title in visited:
            queue.task_done()
            continue
        
        visited.add(page_title)
        print(f"Processing: {page_title}")  # Debug output
        page = wiki_wiki.page(page_title)
        if not page.exists():
            print(f"Page does not exist: {page_title}")  # Debug output
            queue.task_done()
            continue
        
        text = page.text
        print(f"Text extracted: {text[:100]}...")  # Debug output
        
        if classify_text(text):
            with lock:
                if saved_count < MAX_PAGES:
                    file_path = os.path.join(DATASET_DIR, f"{page_title}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    saved_count += 1
                    print(f"Saved: {file_path} ({saved_count}/{MAX_PAGES})")  # Debug output with count
        
        # Add subcategory links to the queue
        for link in page.links.keys():
            if link not in visited:
                queue.put(link)
        
        queue.task_done()

# Start concurrent processing
num_threads = 8
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    for _ in range(num_threads):
        executor.submit(process_page)

queue.join()
print(f"Scraping completed. {saved_count} pages saved.")