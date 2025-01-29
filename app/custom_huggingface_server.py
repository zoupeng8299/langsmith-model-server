import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.custom_huggingface_chat_model import CustomChatModel
from app.custom_huggingface_model import CustomLLM
import logging
from langserve import add_routes

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
    "deepseek_qwen_7b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "local_path": "/Users/zoupeng/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-v0.3",
        "local_path": "/Users/zoupeng/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3",
    },
    "deepseek_qwen_1.5b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "local_path": "/Users/zoupeng/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
    },
    "opt": {
        "name": "facebook/opt-125m",
        "local_path": "/Users/zoupeng/.cache/huggingface/hub/models--facebook--opt-125m",
    }
}

MODEL_OPTIONS = {
    1: "deepseek_qwen_7b",
    2: "mistral",
    3: "deepseek_qwen_1.5b",
    4: "opt"
}

def select_model():
    print("Select the model to use:")
    for key, value in MODEL_OPTIONS.items():
        print(f"{key}: {value}")
    choice = int(input("Enter the number of the model: "))
    model_key = MODEL_OPTIONS.get(choice)
    if model_key is None:
        raise ValueError("Invalid model selection")
    
    # 检查是否为 deepseek 模型
    include_think_process = False
    if "deepseek" in model_key.lower():
        think_choice = input("Do you want to include thinking process? (y/n): ").lower()
        include_think_process = think_choice.startswith('y')
    
    return model_key, include_think_process

if __name__ == "__main__":
    model_key, include_think_process = select_model()
    model_config = MODELS[model_key]
    llm_name = model_config["name"]
    
    logger.info(f"Starting server with model: {llm_name}")
    logger.info(f"Include thinking process: {include_think_process}")
    
    # !!! The with_configurable_fields may cause reload llm, be careful to use it 
    # configurable_chat_model = CustomChatModel(llm_name=llm_name).with_configurable_fields() if hasattr(CustomChatModel, 'with_configurable_fields') else CustomChatModel(llm_name=llm_name)
    configurable_chat_model = CustomChatModel(
        llm_name=llm_name,
        include_think_process=include_think_process
    )
    add_routes(app, configurable_chat_model, path="/chat")

    # configurable_llm = CustomLLM().with_configurable_fields() if hasattr(CustomLLM, 'with_configurable_fields') else CustomLLM()
    configurable_llm = CustomLLM()
    add_routes(app, CustomLLM())

    uvicorn.run(app, host="0.0.0.0", port=7302)