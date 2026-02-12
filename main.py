import sys
import os
from langchain_ollama import ChatOllama
import torch

def check_cuda():
    print("--- Checking CUDA ---")
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is NOT available. Running on CPU.")
    print("---------------------")

def check_ollama():
    print("\n--- Checking Ollama Connection ---")
    try:
        llm = ChatOllama(model="llama3") 
        response = llm.invoke("Hello, are you ready to help me build a RAG system?")
        print(f"Ollama Response:\n{response.content}")
        print("----------------------------------")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running ('ollama serve') and you have pulled the model (e.g., 'ollama pull llama3.1').")

if __name__ == "__main__":
    check_cuda()
    check_ollama()
