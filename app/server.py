from app.custom_chat_model import CustomChatModel
from app.custom_model import CustomLLM
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()

configurable_chat_model = CustomChatModel().with_configurable_fields() if hasattr(CustomChatModel, 'with_configurable_fields') else CustomChatModel()
add_routes(app, configurable_chat_model, path="/chat")

configurable_llm = CustomLLM().with_configurable_fields() if hasattr(CustomLLM, 'with_configurable_fields') else CustomLLM()
add_routes(app, CustomLLM())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="192.168.1.100", port=11434)
