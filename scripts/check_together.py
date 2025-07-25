import os
from dotenv import load_dotenv
from together import Together

load_dotenv()

client = Together()  # Auth picks up TOGETHER_API_KEY from env

response = client.chat.completions.create(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    messages=[{"role": "user", "content": "What are some fun things to do in Sydney?"}],
)

print(response.choices[0].message.content)
