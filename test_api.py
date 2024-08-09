import requests
import os
from dotenv import load_dotenv

url = "http://localhost:8000/recommend"
headers = {"Content-Type": "application/json"}
data = {"title": "spiderman", "n_recommendations": 20}

response = requests.post(url, headers=headers, json=data)
print(response.json())

load_dotenv()
api_key = os.getenv("API_KEY")
print(api_key)


