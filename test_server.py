from langserve import RemoteRunnable

chat_model = RemoteRunnable("http://192.168.1.100:11434/chat")

print(f"Pinging chat model: {chat_model.invoke('help').content}")

instruct_model = RemoteRunnable("http://192.168.1.100:11434")
print(f"Pinging instruct model: {instruct_model.invoke('help')}")
