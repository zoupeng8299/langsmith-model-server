from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessageChunk, BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import ConfigurableField, Runnable


class CustomChatModel(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    This model returns the first n characters of the input prompt. This is intended to serve as a template for
    your own chat style model you may want to expose in the playground.
    """

    n: int = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm = ChatOllama(
            model="deepseek-r1:8b",
            base_url="http://192.168.1.100:11434",
            temperature=0,
            config={
                "num_ctx": 8192  # safe context length
            }
        )

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
        response = self.llm.generate(messages, stop=stop)
        message = AIMessage(
            content=response['content'],
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                "time_in_seconds": response['time_in_seconds'],
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

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
        response = self.llm.stream(messages, stop=stop)
        for chunk in response:
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk['content']))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": response['time_in_seconds']})
        )

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
        return self.configurable_fields(n=ConfigurableField(
            id="n",
            name="Num Characters",
            description="Number of characters to return from the input prompt.",
        ))

