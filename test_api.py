# curl -X POST -H "Content-Type: application/json" -d '{"title": "Toy Story", "n_recommendations": 15}' http://localhost:8000/recommend 
import requests

url = 'http://localhost:8000/recommend'

data = {"title": "Toy Story", "n_recommendations": 15}



