from pydoc import text
import re

import numpy as np
import pandas as pd

def load_data(file_path):
    """
    Load data from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_json(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
data = load_data('data/raw/arabic_fake_news_dataset.json')

# now we apply explode() to the 'text' column to split the text into individual words

fake_df = data[["fakes"]].explode("fakes")
fake_df.columns = ["text"]
fake_df["label"] = 0

# now we apply explode() to the 'true' column to split the text into individual words
true_df = data[['trues']].explode("trues")
true_df.columns = ["text"]
true_df["label"] = 1

# now we concatenate the two dataframes to create a single dataframe with both fake and true news
final_df = pd.concat([fake_df, true_df], ignore_index=True)

# print(final_df.shape)
# print(final_df.head(10))
# print(final_df['label'].value_counts())

# now we drop link rows from the text column
final_df = final_df[~final_df['text'].str.contains('http')]

# now we remove some special characters from the text column  الإدعاء  \n withe .str.replace()
final_df['text'] = final_df['text'].str.replace(r'الإدعاء|\n', ' ', regex=True)


def clean_text(text): 
    """
    Clean the input text by removing special characters and extra spaces.

    Parameters:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    # Remove non-Arabic characters
    if not isinstance(text, str):  # ← هنا مباشرة
        return ""
    # Remove special characters and digits
    cleaned_text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)  # شيل التشكيل
    cleaned_text = re.sub(r'[أإآ]', 'ا', cleaned_text)  # normalize الهمزات
    # Remove punctuation   
    # Remove numbers
    cleaned_text = re.sub(r'\d+', '', cleaned_text)  # شيل الأرقام
    cleaned_text = re.sub(r'[^\u0621-\u064A\s]', '', cleaned_text)  # شيل علامات الترقيم   
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# let`s test the function on the first 5 rows of the text column
final_df['text'] = final_df['text'].apply(clean_text)
print(final_df['text'].head(3))

final_df.dropna(subset=['text'], inplace=True)
final_df = final_df[final_df['text'].str.strip() != '']

# save preprocessed data to a new csv file 
final_df.to_csv('data/processed/arabic_fake_news_dataset.csv', index=False)