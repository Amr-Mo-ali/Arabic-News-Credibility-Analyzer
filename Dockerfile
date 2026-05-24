FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('AmrMohamed21/arabert-fake-news'); AutoModelForSequenceClassification.from_pretrained('AmrMohamed21/arabert-fake-news')"

COPY . .

COPY start.sh .
RUN dos2unix start.sh && chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]