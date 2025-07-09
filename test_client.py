#!/usr/bin/env python3
"""
MiniVault API Test Client
A CLI tool for testing the MiniVault API with both regular and streaming requests.
"""

import argparse
import json
import time
import requests
from typing import Dict, Any

class MiniVaultClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text using the regular endpoint"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            result["client_time_ms"] = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def generate_text_stream(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """Generate text using the streaming endpoint"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/generate/stream", 
                json=payload, 
                stream=True
            )
            response.raise_for_status()
            
            print(f"🔄 Streaming response for: '{prompt[:50]}...'")
            print("📝 Response: ", end="", flush=True)
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                            if 'chunk' in data:
                                print(data['chunk'], end="", flush=True)
                                full_response += data['chunk']
                            elif data.get('done'):
                                print(f"\n✅ Complete! Processing time: {data.get('processing_time_ms', 0):.2f}ms")
                                break
                        except json.JSONDecodeError:
                            continue
            
            return {"response": full_response, "streamed": True}
        except Exception as e:
            return {"error": str(e)}
    
    def get_logs(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent logs"""
        try:
            response = self.session.get(f"{self.base_url}/logs", params={"limit": limit})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="MiniVault API Test Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--stream", action="store_true", help="Use streaming endpoint")
    parser.add_argument("--health", action="store_true", help="Check API health")
    parser.add_argument("--logs", action="store_true", help="Show recent logs")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    client = MiniVaultClient(args.url)
    
    # Health check
    if args.health:
        health = client.health_check()
        print("🏥 Health Check:")
        print(json.dumps(health, indent=2))
        return
    
    # Show logs
    if args.logs:
        logs = client.get_logs()
        print("📜 Recent Logs:")
        print(json.dumps(logs, indent=2))
        return
    
    # Interactive mode
    if args.interactive:
        print("🎮 Interactive Mode - Type 'quit' to exit")
        print("Commands:")
        print("  /stream - Toggle streaming mode")
        print("  /health - Check API health")
        print("  /logs - Show recent logs")
        print("  /help - Show this help")
        
        streaming = False
        
        while True:
            try:
                prompt = input("\n💬 Enter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if prompt == '/stream':
                    streaming = not streaming
                    print(f"🔄 Streaming mode: {'ON' if streaming else 'OFF'}")
                    continue
                
                if prompt == '/health':
                    health = client.health_check()
                    print(json.dumps(health, indent=2))
                    continue
                
                if prompt == '/logs':
                    logs = client.get_logs()
                    print(json.dumps(logs, indent=2))
                    continue
                
                if prompt == '/help':
                    print("Commands:")
                    print("  /stream - Toggle streaming mode")
                    print("  /health - Check API health")
                    print("  /logs - Show recent logs")
                    print("  /help - Show this help")
                    continue
                
                if not prompt:
                    continue
                
                # Generate response
                if streaming:
                    client.generate_text_stream(prompt, args.max_tokens, args.temperature)
                else:
                    result = client.generate_text(prompt, args.max_tokens, args.temperature)
                    if "error" in result:
                        print(f"❌ Error: {result['error']}")
                    else:
                        print(f"📝 Response: {result['response']}")
                        print(f"🔧 Model: {result['model_used']}")
                        print(f"⏱️  Time: {result['processing_time_ms']:.2f}ms")
                        print(f"🎯 Tokens: {result['tokens_generated']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return
    
    # Single request mode
    if args.stream:
        result = client.generate_text_stream(args.prompt, args.max_tokens, args.temperature)
    else:
        result = client.generate_text(args.prompt, args.max_tokens, args.temperature)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"📝 Response: {result['response']}")
            if not result.get("streamed"):
                print(f"🔧 Model: {result['model_used']}")
                print(f"⏱️  Time: {result['processing_time_ms']:.2f}ms")
                print(f"🎯 Tokens: {result['tokens_generated']}")

if __name__ == "__main__":
    main()