import requests
import os
import time


url = "http://localhost:8000/recommend"
headers = {"Content-Type": "application/json"}
data = {"title_string": "spiderman", "n_recommendations": 20}
start = time.time()
response = requests.post(url, headers=headers, json=data)
print(response.json())
end = time.time()
print(f" Time took {end - start}")


