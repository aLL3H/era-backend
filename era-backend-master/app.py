from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from transformers import pipeline
from collections import Counter
import sentencepiece

app = Flask(__name__)
CORS(app)

def download_spacy_model():
    try:
        spacy.load("pt_core_news_sm")
    except OSError:
        print("Baixando o modelo pt_core_news_sm...")
        spacy.cli.download("pt_core_news_sm")

download_spacy_model()

nlp = spacy.load("pt_core_news_sm")

summarizer = pipeline("summarization", model="unicamp-dl/ptt5-base-portuguese-vocab")

def extrair_palavras_chave(texto):
    doc = nlp(texto)
    palavras_chave = set()

    for ent in doc.ents:
        palavras_chave.add(ent.text.lower())

    palavras = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    frequencias = Counter(palavras)
    palavras_chave.update([item[0] for item in frequencias.most_common(10)])

    return list(palavras_chave)

def gerar_resumo(texto):
    resumo = summarizer(texto, max_new_tokens=500, do_sample=False)
    return resumo[0]['summary_text']

def identificar_tipo_documento(texto):
    if "saúde" in texto.lower():
        return "Setor Saúde"
    elif "educação" in texto.lower():
        return "Setor Educação"
    elif "finanças" in texto.lower():
        return "Setor Financeiro"
    else:
        return "Geral"

@app.route('/processar-texto', methods=['POST'])
def processar_texto():
    data = request.json
    texto = data.get("texto", "")

    if not texto:
        return jsonify({"erro": "Texto não fornecido"}), 400

    palavras_chave = extrair_palavras_chave(texto)
    resumo = gerar_resumo(texto)
    tipo_documento = identificar_tipo_documento(texto)

    return jsonify({
        "tipo_documento": tipo_documento,
        "palavras_chave": palavras_chave,
        "resumo": resumo
    })

if __name__ == '__main__':
    app.run(debug=True)
