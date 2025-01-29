import uvicorn
from app.custom_ollama_chat_model import CustomChatModel
from app.custom_ollama_model import CustomLLM
from fastapi import FastAPI
from langserve import add_routes
import logging
from fastapi.middleware.cors import CORSMiddleware

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration and mapping
MODELS = {
    "phi4:latest": {
        "name": "phi4:latest",
    },
    "deepseek-r1:14b": {
        "name": "deepseek-r1:14b",
    },
    "dolphin3:latest": {
        "name": "dolphin3:latest",
    },
    "deepseek-r1:8b": {
        "name": "deepseek-r1:8b",
    },
    "mistral:latest": {
        "name": "mistral:latest",
    },
    "qwen2.5:7b": {
        "name": "qwen2.5:7b",
    },
    "lamma3.1:latest": {
        "name": "lamma3.1:latest",
    },
}

MODEL_OPTIONS = {
    1: "phi4:latest",
    2: "deepseek-r1:14b",
    3: "dolphin3:latest",
    4: "deepseek-r1:8b",
    5: "mistral:latest",
    6: "qwen2.5:7b",
    7: "lamma3.1:latest"
    }

def select_model():
    print("Select the model to use:")
    for key, value in MODEL_OPTIONS.items():
        print(f"{key}: {value}")
    choice = int(input("Enter the number of the model: "))
    model_key = MODEL_OPTIONS.get(choice)
    if model_key is None:
        raise ValueError("Invalid model selection")
    return model_key

if __name__ == "__main__":
    model_key = select_model()
    model_config = MODELS[model_key]
    llm_name = model_config["name"]
    
    logger.info(f"Starting server with model: {llm_name}")
    
    # !!! The with_configurable_fields may cause reload llm, be careful to use it 
    # configurable_chat_model = CustomChatModel(llm_name=llm_name).with_configurable_fields() if hasattr(CustomChatModel, 'with_configurable_fields') else CustomChatModel(llm_name=llm_name)
    configurable_chat_model = CustomChatModel(llm_name=llm_name)
    add_routes(app, configurable_chat_model, path="/chat")

    # configurable_llm = CustomLLM().with_configurable_fields() if hasattr(CustomLLM, 'with_configurable_fields') else CustomLLM()
    configurable_llm = CustomLLM()
    add_routes(app, CustomLLM())

    uvicorn.run(app, host="0.0.0.0", port=7301)
