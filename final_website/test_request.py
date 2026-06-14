import requests, json

url = "http://127.0.0.1:8000/predict"
payload = {
    "question": "Test question",
    "answer": "This is a normal medical answer."
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print("Status Code:", response.status_code)
print("Response Body:", response.text)
