import requests

# YOUR EC2 instance URL
url = "http://3.238.89.41:8000/upload/"

# 🔥 VIDEO FILE (LOCAL)
file_path = "video.mp4"

with open(file_path, "rb") as f:
    files = {"file": f}
    
    response = requests.post(url, files=files)

print(response.json())
