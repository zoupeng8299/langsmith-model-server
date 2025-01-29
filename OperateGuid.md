## We provide two type service: ollama and huggingface
* Ollma server runs on Macmini M4 ladtap by controlling some about 7 llms
* HuggingFace control about 5 llms which run Macmini M4 computer

### Ollama service
Ollama service was provided on Macmini and MacbookPro m1, you can run client to request both service. Because OllamaChat can access url api that link the service.

### HuggingFace service
HugggingFace servcie runs together with llms, so it runs only on Macmini, can not run another computer. Because HuggingFacePipeline is local, only access local llms. It was encapsolated by HuggingFaceChat which from langchain base Chatmodel. So it can invoke messages and so on.

### How to deploy and run 

On any computer in local network, you can run the client.

On any computer in local network, you can provide ollama service, because ollama server provide url api service.

You only run huggingface service on the server computer, because they must together.

1. For ollama, you must run ollama server first, then run the service.
2. For huggingface, you only run the service, the service will load llm.

For example:

```python custom_ollama_server.py```

```python custom_huggingface_server.py```

3. At last, you can run client in any computer.

For example: 

```python test_ollama_server.py```

*Note:* you must run on right enviroment!!!