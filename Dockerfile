FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Pre-download the model at build time
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('AmrMohamed21/arabert-fake-news'); AutoModelForSequenceClassification.from_pretrained('AmrMohamed21/arabert-fake-news')"

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the application will run on
EXPOSE  7860

# Command to run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
