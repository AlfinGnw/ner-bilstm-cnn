from flask import Flask, render_template, request, flash
import pickle
import numpy as np
from tensorflow import keras
import re
from newspaper import Article
from tensorflow.keras.preprocessing.text import text_to_word_sequence

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Load model dan resources
model = keras.models.load_model('bilstm_cnn_ner(1).keras')

with open('word_vocab(1).pkl', 'rb') as f:
    word_vocab = pickle.load(f)

with open('label_vocab(1).pkl', 'rb') as f:
    label_vocab = pickle.load(f)

# Reverse label vocabulary
index_to_label = {v: k for k, v in label_vocab.items()}

# Fungsi normalisasi (sama seperti model)
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

    # Format tanggal dalam tanda kurung: (21/12/2022) -> 21/12/2022
    text = re.sub(r'\((\d{1,2}/\d{1,2}/\d{4})\)', r'\1', text)

    # Format 'dd Month yyyy'
    pattern_dd_month_yyyy = r'\b(\d{1,2})\s+(' + '|'.join(month_to_num.keys()) + r')\s+(\d{4})\b'
    text = re.sub(
        pattern_dd_month_yyyy,
        lambda m: f"{m.group(1)}/{month_to_num[m.group(2).lower()]}/{m.group(3)}",
        text,
        flags=re.IGNORECASE
    )

    # Format 'yyyy-mm-dd'
    text = re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', r'\3/\2/\1', text)

    # Format 'dd/mm/yyyy' atau 'dd/mm/yy'
    text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', r'\1/\2/\3', text)

    # Format lain: dd-mm-yyyy, dd.mm.yyyy
    text = re.sub(r'\b(\d{1,2})[-.](\d{1,2})[-.](\d{4})\b', r'\1/\2/\3', text)

    return text

# Fungsi tokenisasi dengan TensorFlow (sama seperti model)
def tokenize(text):
    return text_to_word_sequence(text, filters='!"#$%&()*+,-.:;<=>?@[\\]^_`{|}~\t\n')

def predict_entities(text, max_seq_len=512):
    # Normalisasi dan tokenisasi
    normalized_text = normalize_text(text)
    tokens = tokenize(normalized_text)
    
    # Konversi ke indeks
    tokens_idx = [word_vocab.get(word, word_vocab["<UNK>"]) for word in tokens]
    
    # Padding
    padded_tokens = keras.preprocessing.sequence.pad_sequences(
        [tokens_idx], maxlen=max_seq_len, padding="post", value=0
    )
    
    # Prediksi
    predictions = model.predict(padded_tokens)
    predicted_labels = np.argmax(predictions, axis=-1)[0]
    
    # Mapping label
    return [
        (token, index_to_label[label_idx]) 
        for token, label_idx in zip(tokens, predicted_labels[:len(tokens)])
    ]

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
                article = Article(url)
                article.download()
                article.parse()
                scraped_text = article.text
                
                if not scraped_text:
                    flash('Tidak dapat mengambil konten dari URL tersebut. Konten kosong.', 'danger')
                    
            except Exception as e:
                flash(f"Error scraping URL: {str(e)}", 'danger')
                scraped_text = ''
                
        elif action == 'detect':
            text = request.form.get('text', '')
            if text:
                # Prediksi entitas
                predicted = predict_entities(text)
                
                # Grouping entitas
                entity_dict = {}
                current_entity = []
                current_label = None
                
                for token, label in predicted:
                    if label == 'O':
                        if current_entity:
                            entity_type = current_label.split('-')[-1] if '-' in current_label else current_label
                            entity_text = ' '.join(current_entity)
                            
                            # Hanya tambahkan jika entitas belum ada dalam daftar (menghilangkan duplikat)
                            if entity_type not in entity_dict or entity_text not in entity_dict[entity_type]:
                                entity_dict.setdefault(entity_type, []).append(entity_text)
                                
                            current_entity = []
                            current_label = None
                    else:
                        if '-' in label:
                            prefix, entity_type = label.split('-', 1)
                        else:
                            prefix, entity_type = 'B', label  # Default prefix jika tidak ada tanda '-'
                        
                        if prefix == 'B' or current_label != entity_type:
                            # Jika ada entitas yang sedang diproses, simpan dahulu
                            if current_entity:
                                entity_text = ' '.join(current_entity)
                                
                                # Hanya tambahkan jika entitas belum ada dalam daftar
                                if current_label not in entity_dict or entity_text not in entity_dict[current_label]:
                                    entity_dict.setdefault(current_label, []).append(entity_text)
                                    
                            current_entity = [token]
                            current_label = entity_type
                        else:  # Jika prefix 'I' dan label sama, lanjutkan entitas saat ini
                            current_entity.append(token)
                
                # Handle last entity
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    
                    # Hanya tambahkan jika entitas belum ada dalam daftar
                    if current_label not in entity_dict or entity_text not in entity_dict[current_label]:
                        entity_dict.setdefault(current_label, []).append(entity_text)
                
                entities = entity_dict
                scraped_text = text

    return render_template(
        'index.html',
        entities=entities,
        scraped_text=scraped_text,
        url=url
    )

if __name__ == '__main__':
    app.run(debug=True)