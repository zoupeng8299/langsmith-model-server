from typing import Any, Dict, Iterator, List, Optional
from pydantic import PrivateAttr
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import AIMessageChunk, BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import ConfigurableField, Runnable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import psutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)

def get_device():
    """Detect the available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class CustomChatModel(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    This model returns the first n characters of the input prompt. This is intended to serve as a template for
    your own chat style model you may want to expose in the playground.
    """

    n: int = 5
    # set default llm name id
    llm_name: str = "mistralai/Mistral-7B-v0.3"
    include_think_process: bool = False
    _llm: ChatHuggingFace = PrivateAttr()
    _tokenizer: AutoTokenizer = PrivateAttr()
    _model: AutoModelForCausalLM = PrivateAttr()

    def __init__(self, llm_name: str, **kwargs):
        
        logger.info("CustomChatModel init called")
        
        super().__init__(**kwargs)
        """ 
        repo_id指的是Hugging Face 云端的模型，而不是本地模型。
        HuggingFaceEndpoint 类用于通过 Hugging Face 的 API 访问云端模型。
        ChatHuggingFace 类用于将 HuggingFaceEndpoint 类封装为一个 ChatHuggingFace 实例。
        如果需要使用本地模型，可以使用 HuggingFacePipeline 类, 同样用ChatHuggingFace 封装。
        https://python.langchain.com/docs/integrations/chat/huggingface/
        """
        """
        self._llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=self.llm_name,
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            ),
            verbose=True
        )
        """
        self.llm_name = llm_name
        
        logger.info(f"Loading tokenizer for {self.llm_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.llm_name, 
            trust_remote_code=True
        )

        # get device
        device = get_device()
        logger.info("=== System Information ===")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"Device: {device}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"MPS Available: {torch.backends.mps.is_available()}")
        logger.info(f"RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
        logger.info("=== Load Information ===")
        logger.info(f"Loading model on {device}...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None
        ).to(device)

        logger.info(f"Creating ChatHuggingFace ...")
        hf_pipeline = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer)
        self._llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=hf_pipeline))


    def _clean_response(self, content: str) -> str:
        """清理响应中的特殊标记及其包含的内容"""
        # 如果响应以 <｜begin▁of▁sentence｜> 开头，找到最后一个 <｜Assistant｜> 并从那里开始截取
        if "<｜begin▁of▁sentence｜>" in content:
            try:
                last_assistant_pos = content.rindex("<｜Assistant｜>")
                content = content[last_assistant_pos + len("<｜Assistant｜>"):]
            except ValueError:
                # 如果找不到标记，返回原始内容
                pass
        
        # 清理其他可能残留的标记
        content = (content.replace("<｜begin▁of▁sentence｜>", "")
                        .replace("<｜User｜>", "")
                        .replace("<｜Assistant｜>", "")
                        .strip())
        
        logger.debug(f"Cleaned content: {content}")
        return content

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        logger.info(f"Generating response for messages: {messages}")
        
        """
        # 将消息转换为 HuggingFace 支持的格式
        hf_messages = messages

        response = self._llm.invoke(hf_messages)
        message_content = response['content'] if isinstance(response, dict) else response.content
        message = AIMessage(
            content=message_content,
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                "model": self._llm.llm.repo_id,
                "max_new_tokens": 512,
                "do_sample": False,
                "repetition_penalty": 1.03
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
        """
        
        try:
            # 直接调用封装的 ChatHuggingFace 实例
            response = self._llm.invoke(
                messages,
                max_new_tokens=1024,
                min_length=200,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            
            message_content = response.content if isinstance(response, AIMessage) else response.get('content', '')
            logger.info(f"Response content: {message_content}")
            
            # 清理特殊标记
            message_content = self._clean_response(message_content)
            logger.info(f"Cleaned response content: {message_content}")
            # 分离思考过程和实际响应
            think_content = ""
            final_content = message_content
            
            # 提取思考过程
            if "<think>" in message_content and "</think>" in message_content:
                think_parts = message_content.split("</think>")
                if len(think_parts) > 1:
                    think_content = think_parts[0].replace("<think>", "").strip()
                    final_content = think_parts[1].strip()
            
            # 根据配置决定是否包含思考过程
            if self.include_think_process:
                logger.info(f"Including thinking process: {think_content}")
                final_content = f"Thinking process:\n{think_content}\n\nResponse:\n{final_content}"
            
            message = AIMessage(
                content=final_content,
                additional_kwargs={
                    "think_process": think_content if think_content else None
                },
                response_metadata={
                    "model": self.llm_name,
                    "max_new_tokens": 1024,
                    "min_length": 200,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9
                }
            )
            
            return ChatResult(generations=[ChatGeneration(message=message)])
                
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise
    
    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method. Do not implement this method if the model
        does not support streaming.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        
        try:
            response = self._llm.stream(
                messages,
                max_new_tokens=1024,
                min_length=200,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )

            # The following code is DEMO for the stream response
            """
            for chunk in response:
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk['content']))
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", response_metadata={"time_in_sec": response['time_in_seconds']})
            )
            """
            
            # The following code is for the thinking process and response content
            # But I don't use it in this model for no details in the response
            """
            in_think_block = False
            think_content = []
            response_content = []
        
            for chunk in response:
                # 检查 chunk 的类型并正确提取内容
                if isinstance(chunk, AIMessage):
                    content = chunk.content
                else:
                    content = chunk.get('content', '')
                
                # 清理特殊标记
                content = self._clean_response(content)

                # 处理思考块的开始和结束
                if "<think>" in content:
                    in_think_block = True
                    content = content.replace("<think>", "")
                    if self.include_think_process:
                        logger.info(f"Including thinking process: {content}")
                        yield ChatGenerationChunk(message=AIMessageChunk(content="Thinking process:\n"))
                elif "</think>" in content:
                    in_think_block = False
                    content = content.replace("</think>", "")
                    if self.include_think_process:
                        logger.info(f"Including thinking process: {content}")
                        yield ChatGenerationChunk(message=AIMessageChunk(content="\nResponse:\n"))
                    continue
                
                # 根据是否在思考块中和配置来决定是否输出内容
                if in_think_block:
                    think_content.append(content)
                    if self.include_think_process:
                        logger.info(f"Including thinking process: {content}")
                        yield ChatGenerationChunk(message=AIMessageChunk(content=content))
                else:
                    response_content.append(content)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=content))
            
            # 添加元数据
            metadata = {
                "model": self.llm_name,
                "think_process": "".join(think_content),
                "response_length": len("".join(response_content))
            }
            
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", response_metadata=metadata)
            )
            """        
            for chunk in response:
                # 检查 chunk 的类型并正确提取内容
                if isinstance(chunk, AIMessage):
                    content = chunk.content
                else:
                    content = chunk.get('content', '')
                
            yield ChatGenerationChunk(message=AIMessageChunk(content=content))

            # 在流式响应结束时添加元数据
            metadata = {"model": self.llm_name}
            if hasattr(response, 'time_in_seconds'):
                metadata["time_in_sec"] = response.time_in_seconds
                
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", response_metadata=metadata)
            )
            
        except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                raise
        

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "custom-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    def with_configurable_fields(self) -> Runnable:
        """Expose fields you want to be configurable in the playground. We will automatically expose these to the
        playground. If you don't want to expose any fields, you can remove this method."""
        return self.configurable_fields(llm_name=ConfigurableField(
            id="llm_name",
            name="LLM Name",
            description="LLM's name which ollama holds.",
        ))