import openai
import os
from openai.embeddings_utils import get_embedding

with open('./api_key.txt', 'r') as f:
    OPENAI_API_KEY = f.readline().strip()
openai.api_key = OPENAI_API_KEY
os.environ["http_proxy"]="127.0.0.1:50918"
os.environ["https_proxy"]="127.0.0.1:50918"

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    if response.choices and response.choices[0].message["content"]:
        return response.choices[0].message["content"]
    else:
        return "No valid response received."


text = """
你应该提供尽可能清晰、具体的指示，以表达你希望模型执行的任务。\
这将引导模型朝向所需的输出，并降低收到无关或不正确响应的可能性。\
不要将写清晰的提示与写简短的提示混淆。\
在许多情况下，更长的提示可以为模型提供更多的清晰度和上下文信息，从而导致更详细和相关的输出。
"""

prompt = """
通过原生的js为我实现一段代码，监听浏览器url的变化，并且将其打印出来
"""

# print(prompt)
response = get_completion(prompt)
print(response)

# 输出向量
embedding = get_embedding('我爱吃苹果', engine="text-embedding-ada-002")
print(embedding)