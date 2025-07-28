from flask import Flask, render_template, request, flash
import pickle
import numpy as np
from tensorflow import keras
import re
from newspaper import Article, Config
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

app = Flask(__name__)
app.secret_key = 'phantom'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

SELENIUM_OPTIONS = Options()
SELENIUM_OPTIONS.add_argument(f'user-agent={USER_AGENT}')
SELENIUM_OPTIONS.add_argument('--headless')
SELENIUM_OPTIONS.add_argument('--disable-gpu')
SELENIUM_OPTIONS.add_argument('--no-sandbox')
SELENIUM_OPTIONS.add_argument('--disable-dev-shm-usage')

CHROME_DRIVER_PATH = '/path/to/chromedriver.exe'

model = keras.models.load_model('bilstm_cnn_ner (2).keras')

with open('word_vocab (2).pkl', 'rb') as f:
    word_vocab = pickle.load(f)

with open('label_vocab (2).pkl', 'rb') as f:
    label_vocab = pickle.load(f)

index_to_label = {v: k for k, v in label_vocab.items()}

def normalize_text(text):
    text = text.lower()
    text = normalize_dates(text)
    text = re.sub(r"[^\w\s/]", "", text)
    return text.strip()

def normalize_dates(text):
    month_to_num = {
        'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
        'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
        'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
    }

    text = re.sub(r'\((\d{1,2}/\d{1,2}/\d{4})\)', r'\1', text)

    pattern_dd_month_yyyy = r'\b(\d{1,2})\s+(' + '|'.join(month_to_num.keys()) + r')\s+(\d{4})\b'
    text = re.sub(
        pattern_dd_month_yyyy,
        lambda m: f"{m.group(1)}/{month_to_num[m.group(2).lower()]}/{m.group(3)}",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', r'\3/\2/\1', text)
    text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', r'\1/\2/\3', text)
    text = re.sub(r'\b(\d{1,2})[-.](\d{1,2})[-.](\d{4})\b', r'\1/\2/\3', text)

    return text

def tokenize(text):
    return text_to_word_sequence(text, filters='!"#$%&()*+,-.:;<=>?@[\\]^_`{|}~\t\n')

def predict_entities(text, max_seq_len=512):
    normalized_text = normalize_text(text)
    tokens = tokenize(normalized_text)
    tokens_idx = [word_vocab.get(word, word_vocab["<UNK>"]) for word in tokens]
    padded_tokens = keras.preprocessing.sequence.pad_sequences(
        [tokens_idx], maxlen=max_seq_len, padding="post", value=0
    )
    predictions = model.predict(padded_tokens)
    predicted_labels = np.argmax(predictions, axis=-1)[0]
    return [
        (token, index_to_label[label_idx]) 
        for token, label_idx in zip(tokens, predicted_labels[:len(tokens)])
    ]

def extract_with_newspaper(url):
    try:
        config = Config()
        config.browser_user_agent = USER_AGENT
        config.request_timeout = 10
        config.memoize_articles = False
        article = Article(url, config=config)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Newspaper error: {str(e)}")
        return None

def extract_with_selenium(url):
    try:
        service = Service(CHROME_DRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=SELENIUM_OPTIONS)
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        time.sleep(2)
        html = driver.page_source
        driver.quit()
        soup = BeautifulSoup(html, 'html.parser')
        article = soup.find('article')
        if article:
            return article.get_text(separator='\n', strip=True)
        for class_name in ['content', 'article-content', 'post-content', 'entry-content', 'main-content', 'story-content']:
            div = soup.find('div', class_=class_name)
            if div:
                return div.get_text(separator='\n', strip=True)
        paragraphs = soup.find_all('p')
        if paragraphs:
            return '\n'.join(p.get_text(strip=True) for p in paragraphs)
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"Selenium error: {str(e)}")
        return None

def extract_with_bs4(url):
    try:
        headers = {'User-Agent': USER_AGENT}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        article = soup.find('article')
        if article:
            return article.get_text(separator='\n', strip=True)
        for class_name in ['content', 'article-content', 'post-content', 'entry-content', 'main-content', 'story-content']:
            div = soup.find('div', class_=class_name)
            if div:
                return div.get_text(separator='\n', strip=True)
        paragraphs = soup.find_all('p')
        if paragraphs:
            return '\n'.join(p.get_text(strip=True) for p in paragraphs)
        return None
    except Exception as e:
        logger.error(f"BeautifulSoup error: {str(e)}")
        return None

def extract_content(url):
    text = extract_with_newspaper(url)
    if text and text.strip():
        return text
    text = extract_with_bs4(url)
    if text and text.strip():
        return text
    text = extract_with_selenium(url)
    if text and text.strip():
        return text
    return None

def enhance_with_extra_locations(entities, extra_str):
    extra_locations = extra_str.split("|") if extra_str else []
    existing = set(e.lower() for e in entities.get('LOC', []))
    combined = entities.get('LOC', [])[:]
    for loc in extra_locations:
        if loc.lower() not in existing:
            combined.append(loc)
    entities['LOC'] = list(set(combined))
    return entities

@app.route('/', methods=['GET', 'POST'])
def index():
    entities = None
    scraped_text = ''
    url = ''

    if request.method == 'POST':
        url = request.form.get('url', '')
        action = request.form.get('action', '')

        if action == 'scrap' and url:
            try:
                scraped_text = extract_content(url)
                if not scraped_text or not scraped_text.strip():
                    flash('Tidak dapat mengambil konten dari URL tersebut. Mungkin konten dilindungi atau format tidak didukung.', 'danger')
                else:
                    scraped_text = '\n'.join(line.strip() for line in scraped_text.split('\n'))
            except Exception as e:
                logger.error(f"Scraping error: {str(e)}")
                flash(f"Error saat mengambil konten: {str(e)}", 'danger')
                scraped_text = ''

        elif action == 'detect':
            text = request.form.get('text', '')
            if text:
                predicted = predict_entities(text)
                entity_dict = {}
                current_entity = []
                current_label = None
                for token, label in predicted:
                    if label == 'O':
                        if current_entity:
                            ent_text = ' '.join(current_entity)
                            entity_dict.setdefault(current_label, []).append(ent_text)
                            current_entity = []
                            current_label = None
                    else:
                        prefix, label_type = label.split('-', 1) if '-' in label else ('B', label)
                        if prefix == 'B' or current_label != label_type:
                            if current_entity:
                                ent_text = ' '.join(current_entity)
                                entity_dict.setdefault(current_label, []).append(ent_text)
                            current_entity = [token]
                            current_label = label_type
                        else:
                            current_entity.append(token)
                if current_entity:
                    ent_text = ' '.join(current_entity)
                    entity_dict.setdefault(current_label, []).append(ent_text)
                for k in entity_dict:
                    entity_dict[k] = list(set(entity_dict[k]))
                entities = entity_dict
                scraped_text = text

              
                extra_context = request.form.get('lokasi_tambahan', '')
                entities = enhance_with_extra_locations(entities, extra_context)

    return render_template('index.html', entities=entities, scraped_text=scraped_text, url=url)

if __name__ == '__main__':
    app.run(debug=True)
