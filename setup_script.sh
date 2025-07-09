#!/bin/bash

# MiniVault API Setup Script
echo "🚀 Setting up MiniVault API..."

# Create project directory
mkdir -p minivault-api
cd minivault-api

# Create logs directory
mkdir -p logs

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make test client executable
chmod +x test_client.py

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start the API: python app.py"
echo "3. Test with CLI: python test_client.py -p 'Hello world'"
echo "4. Or visit: http://localhost:8000/docs"
echo ""
echo "📋 Optional: Install Ollama for better local LLM support:"
echo "   - Install Ollama: https://ollama.ai/"
echo "   - Pull a model: ollama pull llama2:7b"
echo "   - Restart the API to auto-detect Ollama"