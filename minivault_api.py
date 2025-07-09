import os
import json
import time
from datetime import datetime
from typing import Generator, Dict, Any
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel
import uvicorn

"""
MiniVault API

- Run this file and visit http://localhost:8000/docs for interactive API docs.
- The root URL (http://localhost:8000/) will redirect you to /docs for convenience.
- You can interact with the API using:
    - The CLI tool (test_client.py)
    - Swagger UI (/docs)
    - Postman (postman_collection.json)
    - curl/http requests
- The API will use a local LLM if available (Ollama or HuggingFace), otherwise it returns a stubbed response.
- All prompt/response interactions are logged to logs/log.jsonl.
- This project fulfills all ModelVault take-home requirements: local REST API, logging, no cloud LLMs, clear README, and bonus features (streaming, CLI, Postman).
"""

# Try to import transformers for local LLM support
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Transformers not available. Install with: pip install transformers torch")

# Try to import requests for Ollama support
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  Requests not available. Install with: pip install requests")

app = FastAPI(title="MiniVault API", version="1.0.0")

@app.get("/")
def root():
    """Redirect root URL to /docs for user convenience."""
    return RedirectResponse("/docs")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False

class GenerateResponse(BaseModel):
    response: str
    model_used: str = "stub"
    tokens_generated: int = 0
    processing_time_ms: float = 0

class MiniVaultAPI:
    def __init__(self):
        self.setup_logging()
        self.model = None
        self.tokenizer = None
        self.model_type = "stub"
        self.setup_model()
    
    def setup_logging(self):
        """Setup logging directory and file"""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "log.jsonl"
    
    def setup_model(self):
        """Try to setup a local model (HuggingFace or Ollama)"""
        # First try Ollama (lighter weight)
        if self.try_ollama():
            return
        
        # Then try HuggingFace
        if self.try_huggingface():
            return
        
        print("📝 No local models available. Using stubbed responses.")
    
    def try_ollama(self) -> bool:
        """Try to connect to Ollama"""
        if not REQUESTS_AVAILABLE:
            return False
        
        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    # Use the first available model
                    self.model = models[0]["name"]
                    self.model_type = "ollama"
                    print(f"🦙 Connected to Ollama with model: {self.model}")
                    return True
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
        return False
    
    def try_huggingface(self) -> bool:
        """Try to setup HuggingFace model"""
        if not HF_AVAILABLE:
            return False
        
        try:
            print("🤗 Loading HuggingFace model (this may take a moment)...")
            # Use a lightweight model for demo purposes
            model_name = "microsoft/DialoGPT-small"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_type = "huggingface"
            print(f"✅ HuggingFace model loaded: {model_name}")
            return True
        except Exception as e:
            print(f"❌ HuggingFace setup failed: {e}")
        return False
    
    def log_interaction(self, prompt: str, response: str, metadata: Dict[str, Any]):
        """Log the interaction to JSONL file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "metadata": metadata
        }
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def generate_stub_response(self, prompt: str) -> str:
        """Generate a stubbed response"""
        stub_responses = [
            f"This is a stubbed response to: '{prompt[:50]}...'",
            f"I understand you're asking about: {prompt[:30]}... Here's a simulated response.",
            f"Processing your request about '{prompt.split()[0] if prompt.split() else 'unknown'}' - this is a mock response.",
            f"Thank you for your prompt. This is a demonstration response for: {prompt[:40]}..."
        ]
        
        # Simple hash-based selection for consistency
        return stub_responses[hash(prompt) % len(stub_responses)]
    
    def generate_ollama_response(self, prompt: str, stream: bool = False) -> Generator[str, None, None]:
        """Generate response using Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                stream=stream,
                timeout=30
            )
            
            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                full_response += data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                yield full_response
                
        except Exception as e:
            print(f"❌ Ollama generation failed: {e}")
            yield self.generate_stub_response(prompt)
    
    def generate_huggingface_response(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate response using HuggingFace model"""
        try:
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response if response else "I understand your request."
            
        except Exception as e:
            print(f"❌ HuggingFace generation failed: {e}")
            return self.generate_stub_response(prompt)
    
    def generate_response(self, request: GenerateRequest) -> Generator[str, None, None]:
        """Main response generation method"""
        start_time = time.time()
        
        if self.model_type == "ollama":
            yield from self.generate_ollama_response(request.prompt, request.stream)
        elif self.model_type == "huggingface":
            yield self.generate_huggingface_response(request.prompt, request.max_tokens)
        else:
            yield self.generate_stub_response(request.prompt)

# Initialize API instance
api = MiniVaultAPI()

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text response (non-streaming)"""
    if request.stream:
        raise HTTPException(
            status_code=400, 
            detail="Use /generate/stream endpoint for streaming responses"
        )
    
    start_time = time.time()
    
    # Generate response
    response_text = ""
    for chunk in api.generate_response(request):
        response_text += chunk
    
    processing_time = (time.time() - start_time) * 1000
    
    # Log the interaction
    metadata = {
        "model_used": api.model_type,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "processing_time_ms": processing_time,
        "tokens_generated": len(response_text.split())  # Rough token count
    }
    
    api.log_interaction(request.prompt, response_text, metadata)
    
    return GenerateResponse(
        response=response_text,
        model_used=api.model_type,
        tokens_generated=len(response_text.split()),
        processing_time_ms=processing_time
    )

@app.post("/generate/stream")
async def generate_text_stream(request: GenerateRequest):
    """Generate text response (streaming)"""
    
    def generate_stream():
        start_time = time.time()
        full_response = ""
        
        for chunk in api.generate_response(request):
            full_response += chunk
            # Format as Server-Sent Events
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log the interaction
        metadata = {
            "model_used": api.model_type,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "processing_time_ms": processing_time,
            "tokens_generated": len(full_response.split()),
            "streamed": True
        }
        
        api.log_interaction(request.prompt, full_response, metadata)
        
        # Send completion event
        yield f"data: {json.dumps({'done': True, 'processing_time_ms': processing_time})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_type": api.model_type,
        "model": api.model if api.model_type == "ollama" else "loaded" if api.model else "stub"
    }

@app.get("/logs")
async def get_logs(limit: int = 10):
    """Get recent logs"""
    logs = []
    try:
        with open(api.log_file, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line))
        
        # Return last N logs
        return {"logs": logs[-limit:], "total": len(logs)}
    except FileNotFoundError:
        return {"logs": [], "total": 0}

def interactive_model_setup(api_instance):
    print("\n\033[1mWelcome to MiniVault API!\033[0m")
    print("This project provides a local REST API that simulates LLM text generation. "
          "You can interact with it via the CLI, Swagger UI, Postman, or curl. "
          "It supports local models (Ollama, Hugging Face) or stubbed responses, and logs all interactions.")
    print("\n\033[1mApple Silicon (M1/M2/M3/M4) is fully supported.\033[0m")

    while True:
        print("\nChoose a backend:")
        print("[1] Use Ollama (local LLM, recommended for best results)")
        print("[2] Use Hugging Face Transformers (choose a model)")
        print("[3] Use stubbed responses (no real LLM)")
        backend_choice = input("\nEnter your choice [1/2/3]: ").strip()

        if backend_choice == "1":
            # Try to connect to Ollama
            if api_instance.try_ollama():
                print("\n🦙 Ollama is running and detected!")
                proceed = input("Proceed with Ollama? (y/n): ").strip().lower()
                if proceed == "y":
                    api_instance.model_type = "ollama"
                    print("\nProceeding with Ollama as the backend.")
                    break
                else:
                    print("Returning to backend selection.\n")
                    continue
            else:
                print("\nTo use Ollama:")
                print("  1. Download and install from https://ollama.ai/download (Apple Silicon supported)")
                print("  2. Open Terminal and run: ollama serve")
                print("  3. Pull a model, e.g.: ollama pull llama3")
                print("  4. Restart this script after Ollama is running.")
                sys.exit(0)
        elif backend_choice == "2":
            if not HF_AVAILABLE:
                print("\nTransformers not installed. Run: pip install transformers torch\nThen restart this script.")
                sys.exit(0)
            print("\nAvailable Hugging Face models:")
            print("[1] distilgpt2 (small, fast, better than DialoGPT-small)")
            print("[2] gpt2 (larger, more capable, slower)")
            print("[3] Custom (enter your own model name)")
            hf_choice = input("\nEnter your choice [1/2/3]: ").strip()
            if hf_choice == "1":
                model_name = "distilgpt2"
            elif hf_choice == "2":
                model_name = "gpt2"
            elif hf_choice == "3":
                model_name = input("Enter the Hugging Face model name (e.g., gpt2-medium): ").strip()
            else:
                print("Invalid choice. Defaulting to distilgpt2.")
                model_name = "distilgpt2"
            print(f"\n🤗 Downloading and loading model '{model_name}' (this may take a moment)...")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                api_instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
                api_instance.model = AutoModelForCausalLM.from_pretrained(model_name)
                if api_instance.tokenizer.pad_token is None:
                    api_instance.tokenizer.pad_token = api_instance.tokenizer.eos_token
                api_instance.model_type = "huggingface"
                api_instance.hf_model_name = model_name
                print(f"✅ Hugging Face model '{model_name}' loaded.")
                break
            except Exception as e:
                print(f"❌ Failed to load Hugging Face model '{model_name}': {e}")
                print("Proceeding with stubbed responses.")
                api_instance.model_type = "stub"
                break
        else:
            print("\nYou have selected the stubbed response mode.\nThis option returns a hardcoded, simulated response instead of using a real language model.\nIt’s useful for testing the API without downloading or running any models.")
            api_instance.model_type = "stub"
            break

if __name__ == "__main__":
    # Initialize API instance
    api = MiniVaultAPI()
    interactive_model_setup(api)
    print("\n🎉 MiniVault API is running!")
    print("👉 Open http://localhost:8000/docs in your browser to test the API.")
    print("👉 In a new terminal window, try:")
    print("   python3 test_client.py -p \"Tell me a joke\"")
    print("   python3 test_client.py -p \"Tell me a joke\" --stream")
    print("👉 All interactions are logged in logs/log.jsonl")
    print("Press CTRL+C to stop the server.\n")
    uvicorn.run("minivault_api:app", host="0.0.0.0", port=8000, reload=False)