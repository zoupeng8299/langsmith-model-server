# This client tests the Ollama service on Mac Mini M4
# Ollama LLMs run on Mac Mini M4
import httpx
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langserve import RemoteRunnable

chat_model = RemoteRunnable("http://192.168.1.100:7301/chat")

messages = [
    SystemMessage(content="You are a travel assistant."),
    HumanMessage(content="I'm thinking about taking a trip to Europe."),
    AIMessage(content="Sure! I'd love to help you plan your European adventure."),
    HumanMessage(content="What are some must-see places?")
]

try:
    response = chat_model.invoke(messages)
    print(f"Pinging chat model: {response.content}")
except httpx.HTTPStatusError as e:
    print(f"Error invoking chat model: {e}")

instruct_model = RemoteRunnable("http://192.168.1.100:7301")

messages = [
    SystemMessage(content="You are a travel assistant."),
    HumanMessage(content="Can you suggest a good itinerary for Europe?"),
    AIMessage(content="Absolutely! Visiting Paris, Rome, and Barcelona is a great start."),
    HumanMessage(content="Sounds good. Where should I begin?")
]

try:
    response = instruct_model.invoke(messages)
    print(f"Pinging instruct model: {response}")
except httpx.HTTPStatusError as e:
    print(f"Error invoking instruct model: {e}")