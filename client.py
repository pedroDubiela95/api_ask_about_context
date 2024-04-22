import requests
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

# Uploads (<- Post)
url = 'http://127.0.0.1:5000/uploads'
file = open('precos.jpg', 'rb')
data   = {"openai_api_key":OPENAI_API_KEY}
files  = {"file": file}
output = requests.post(url, data=data, files=files)
output.json()


# Query (-> Get)
url = 'http://127.0.0.1:5000/query'
data={
        "openai_api_key":OPENAI_API_KEY,
        "query": """
            Quais são os carros disponíveis?; 
            Quais são os carros listados em 2016?
            """
    }
output = requests.get(url, data=data)
result = output.json()
print(result)



