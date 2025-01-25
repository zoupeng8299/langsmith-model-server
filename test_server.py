from langserve import RemoteRunnable

chat_model = RemoteRunnable("http://192.168.1.57:5001/chat")

print(f"Pinging chat model: {chat_model.invoke('help').content}")

instruct_model = RemoteRunnable("http://192.168.1.57:5001")
print(f"Pinging instruct model: {instruct_model.invoke('help')}")
