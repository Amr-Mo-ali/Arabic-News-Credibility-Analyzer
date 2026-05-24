import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

base_url = "https://www.misbar.com"
headers = {'User-Agent': 'Mozilla/5.0'}

# جيب عنوان من خبر واحد
test_url = base_url + "/factcheck/2026/05/23/%D8%A7%D9%84%D9%81%D9%8A%D8%AF%D9%8A%D9%88-%D9%84%D9%8A%D8%B3-%D9%84%D8%A7%D8%B3%D8%AA%D9%87%D8%AF%D8%A7%D9%81-%D8%AD%D8%B2%D8%A8-%D8%A7%D9%84%D9%84%D9%87-%D9%85%D9%82%D8%B1%D9%91%D9%8B%D8%A7-%D8%A5%D8%B3%D8%B1%D8%A7%D8%A6%D9%8A%D9%84%D9%8A%D9%91%D9%8B%D8%A7-%D8%B3%D8%B1%D9%8A%D9%91%D9%8B%D8%A7"
response = requests.get(test_url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# جرب تجيب العنوان
title = soup.find('h1')
print(title.text.strip() if title else "مش لاقي h1")