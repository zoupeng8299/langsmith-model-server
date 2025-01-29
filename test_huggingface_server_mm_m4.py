# this lient test the huggingface service on macmini m4
# Huggingface llms run on macmini m4
from langserve import RemoteRunnable
import httpx

chat_model = RemoteRunnable("http://192.168.1.100:7302/chat")

try:
    response = chat_model.invoke('I plan to teach AI? think step by step.')
    print(f"Pinging chat model: {response.content}")
except httpx.HTTPStatusError as e:
    print(f"Error invoking chat model: {e}")

instruct_model = RemoteRunnable("http://192.168.1.100:7302")
try:
    response = instruct_model.invoke('How about USA?')
    print(f"Pinging instruct model: {response}")
except httpx.HTTPStatusError as e:
    print(f"Error invoking instruct model: {e}")