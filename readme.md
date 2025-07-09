# MiniVault API

MiniVault API lets you send a text prompt and get a generated response, simulating a local AI model. You can interact with it using the CLI, curl, Postman, or review interactive API docs in your browser (for testing endpoints).

This is all intended to run in the command line interface, streaming tokens.

It first detects if any models are already installed then gives you options to select from a few models you'd like to use to run all locally, or you can start with a demo or "stubbed" version until you decide to install a model.

## 🚀 Quick Start

```bash
# 1. Clone the project from GitHub
git clone https://github.com/jimbrend/MiniVault.git

Then navigate to the folder, by default on Mac this will be in your home directory so you can just type:
cd MiniVault

# 2. From the new project directory, in MiniVault, reate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements_txt.txt
(or usually pip3 will work instead on macOS)
pip3 install -r requirements_txt.txt

# 4. Start the API
python3 minivault_api.py


# 4. Now you can open a new terminal and recommended to first run:
source venv/bin/activate

(this is recommended to keep it self-contained from system-wide Python packages)

Example commands after you've gone through the installation selections:
python3 test_client.py -p "What is the meaning of life?" --stream

add streaming flag to see streaming response:
python3 test_client.py -p "What is the meaning of life?" --stream

```

On first run, you’ll see a prompt to install a local model (Ollama or Hugging Face), or you can continue with stubbed responses.

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs` (for API testing only).

---

## How to Use

You can interact with the API in several ways (make sure the server is running):

### **A. CLI Tool (Recommended)**
```bash
python3 test_client.py -p "Tell me a joke"
python3 test_client.py --interactive
```

### **B. Browser (Swagger UI)**
Go to [http://localhost:8000/docs](http://localhost:8000/docs)
to view endpoints and check health etc.

### **C. curl**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?"}'
```

### **D. Postman**
Import `postman_collection.json` and use the pre-configured requests.

---

## 📋 Features

### Core Functionality
- ✅ **POST /generate** - Generate text responses
- ✅ **Streaming support** - Real-time token-by-token output
- ✅ **JSONL logging** - All interactions logged to `logs/log.jsonl`
- ✅ **Health checks** - API status and model info

### Optional Enhancements (All Implemented!)
- 🤖 **Local LLM Integration** - Supports both Ollama and HuggingFace
- 🌊 **Streaming responses** - Token-by-token generation
- 🧪 **CLI testing tool** - Interactive command-line client
- 📬 **Postman collection** - Ready-to-use API collection
- 📊 **Comprehensive logging** - Detailed interaction tracking

## 🔧 API Endpoints

### POST /generate
Generate text response (non-streaming)

**Request:**
```json
{
  "prompt": "Tell me a story about AI",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "Once upon a time, in a digital realm...",
  "model_used": "ollama",
  "tokens_generated": 45,
  "processing_time_ms": 1250.5
}
```

### POST /generate/stream
Generate streaming text response

**Request:** Same as above with optional `"stream": true`

**Response:** Server-Sent Events stream with chunks

### GET /health
Check API health and model status

### GET /logs
Retrieve recent interaction logs

## 🎮 Testing Tools

### 1. CLI Client
Interactive command-line tool for testing:

```bash
# Single request
python test_client.py -p "Hello world"

# Streaming request
python test_client.py -p "Tell me a joke" --stream

# Interactive mode
python test_client.py --interactive
```

**Interactive Commands:**
- `/stream` - Toggle streaming mode
- `/health` - Check API health
- `/logs` - Show recent logs
- `/help` - Show available commands

### 2. Postman Collection
Import `MiniVault_API.postman_collection.json` into Postman for GUI testing.

## 🤖 Local LLM Support

The API automatically detects and uses local LLMs in this priority order:

### 1. Ollama (Recommended)
- **Install:** [https://ollama.ai/](https://ollama.ai/)
- **Setup:** `ollama pull llama2:7b` (or any model)
- **Auto-detection:** API connects automatically when Ollama is running

### 2. HuggingFace Transformers
- **Auto-setup:** Uses `microsoft/DialoGPT-small` if available
- **Requirements:** `transformers` and `torch` (included in requirements.txt)

### 3. Fallback: Stubbed Responses
- **Always available:** Generates realistic mock responses
- **No setup required:** Works out of the box

## 📝 Logging

All interactions are logged to `logs/log.jsonl` with:

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "prompt": "User's input prompt",
  "response": "Generated response",
  "metadata": {
    "model_used": "ollama",
    "max
