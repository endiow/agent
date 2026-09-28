import requests

# 调用你本地的 Ollama
url = "http://localhost:11434/api/generate"
data = {
    "model": "qwen2.5:3b",
    "prompt": "用一句话解释什么是人工智能",
    "stream": False
}

response = requests.post(url, json=data)
print(response.json()["response"])