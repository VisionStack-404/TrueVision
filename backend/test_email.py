import requests

payload = {
    "name": "Test User",
    "email": "varunvadlakonda1577@gmail.com",
    "is_new_user": True
}
try:
    res = requests.post("http://localhost:8000/send-welcome-email/", json=payload)
    print(res.status_code)
    print(res.text)
except Exception as e:
    print(e)
