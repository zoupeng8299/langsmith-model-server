# this client test ollama servic on macbook pro m1
# In fact, the ollama llms run on macmini m4
from langserve import RemoteRunnable
import httpx

chat_model = RemoteRunnable("http://192.168.1.57:5001/chat")

try:
    response = chat_model.invoke('hello')
    print(f"Pinging chat model: {response.content}")
except httpx.HTTPStatusError as e:
    print(f"Error invoking chat model: {e}")

instruct_model = RemoteRunnable("http://192.168.1.57:5001")
try:
    response = instruct_model.invoke('how about USA?')
    print(f"Pinging instruct model: {response}")
except httpx.HTTPStatusError as e:
    print(f"Error invoking instruct model: {e}")