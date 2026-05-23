# 🔍 Arabic News Credibility Analyzer

<div align="center">

![Arabic NLP](https://img.shields.io/badge/Arabic-NLP-green?style=for-the-badge)
![AraBERT](https://img.shields.io/badge/AraBERT-v2-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-98.8%25-brightgreen?style=for-the-badge)

**An end-to-end Arabic NLP system that detects fake news using Fine-tuned AraBERT — deployed as a production-ready REST API with Docker.**

[🤗 Model on HuggingFace](https://huggingface.co/AmrMohamed21/arabert-fake-news) • [🚀 Live Demo](#how-to-run) • [📊 Results](#results)

</div>

---

## 🎯 Overview

Fake news in Arabic is a critical and largely unsolved problem — **91% of existing fake news research uses English datasets**, leaving the Arabic-speaking world underserved.

This project builds a **production-grade Arabic fake news detector** that:
- Analyzes Arabic news articles and classifies them as **Real** or **Fake**
- Returns a **confidence score** for each prediction
- Exposes a clean **REST API** consumed by an interactive **Streamlit UI**
- Is fully **containerized with Docker** for deployment anywhere

---

## 🏗️ Architecture

```
Arabic News Text
      ↓
Preprocessing Pipeline (Normalization, Diacritics Removal)
      ↓
AraBERT v2 (Fine-tuned on 6,267 Arabic news articles)
      ↓
Classification Head → Fake / Real + Confidence Score
      ↓
FastAPI REST Endpoint → Streamlit UI
      ↓
Docker Compose (Production)
```

---

## ⚡ Tech Stack

| Layer | Technology |
|-------|-----------|
| **NLP Model** | AraBERT v2 (`aubmindlab/bert-base-arabertv2`) |
| **Fine-tuning** | HuggingFace Transformers + Trainer API |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Containerization** | Docker + Docker Compose |
| **Data Processing** | Pandas, Regex (Arabic Unicode) |
| **Model Registry** | HuggingFace Hub |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **98.8%** |
| **F1 Score** | **98.8%** |
| Training Epochs | 3 |
| Dataset Size | 6,267 Arabic articles |
| Train/Test Split | 80/20 |

> Fine-tuned AraBERT significantly outperforms traditional ML baselines (TF-IDF + Logistic Regression) on Arabic fake news detection.

---

## 📁 Project Structure

```
arabic-news-credibility/
│
├── data/
│   ├── raw/              ← Original dataset (not tracked)
│   └── processed/        ← Cleaned dataset (not tracked)
│
├── src/
│   ├── data_prep.py      ← Arabic text preprocessing pipeline
│   ├── train.py          ← AraBERT fine-tuning
│   └── predict.py        ← Inference engine
│
├── api/
│   └── app.py            ← FastAPI REST endpoints
│
├── models/               ← Saved model weights (not tracked)
├── notebooks/            ← EDA and experimentation
├── app_ui.py             ← Streamlit interface
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 How to Run

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/Amr-Mo-ali/arabic-news-credibility.git
cd arabic-news-credibility
docker-compose up --build
```

Open: `http://localhost:8002`

### Option 2 — Local

```bash
# Clone & install
git clone https://github.com/Amr-Mo-ali/arabic-news-credibility.git
cd arabic-news-credibility
pip install -r requirements.txt

# Run API
uvicorn api.app:app --reload

# Run UI (new terminal)
streamlit run app_ui.py
```

### API Usage

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text": "أكد المسؤولون أن الوضع تحت السيطرة"}'
```

**Response:**
```json
{
  "label": "real",
  "confidence": 94.32
}
```

---

## 🗃️ Dataset

- **Source:** [متصدقش (Matsda2sh)](https://matsda2sh.com/) — Egyptian fact-checking platform
- **Size:** 6,267 labeled Arabic news articles
- **Balance:** 3,134 Fake / 3,133 Real (near-perfect balance)
- **Preprocessing:** Diacritics removal, Hamza normalization, punctuation cleaning

---

## 🤗 Model

The fine-tuned model is publicly available on HuggingFace:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("AmrMohamed21/arabert-fake-news")
model = AutoModelForSequenceClassification.from_pretrained("AmrMohamed21/arabert-fake-news")
```

---

## 👨‍💻 Author

**Amr Mohamed**
ML / AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/amr-mohamed21)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Amr-Mo-ali)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://kaggle.com/amrmohammedali)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/AmrMohamed21)