"""
Ollama Model Manager
Manages Ollama models - initialization, health checks, and configuration.
"""

import ollama
import subprocess
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.config import (
    OLLAMA_CONFIG,
    CHAT_MODEL_CONFIG,
    EMBEDDING_MODEL_CONFIG,
    REQUIRED_MODELS,
    PRESETS
)


class OllamaManager:
    """
    Centralized manager for Ollama models.
    Handles model availability, configuration, and health checks.
    """
    
    def __init__(self):
        """Initialize the Ollama manager."""
        self.host = OLLAMA_CONFIG['host']
        self.timeout = OLLAMA_CONFIG['timeout']
        self.chat_model = CHAT_MODEL_CONFIG['model_name']
        self.embedding_model = EMBEDDING_MODEL_CONFIG['model_name']
        
    def check_ollama_running(self) -> bool:
        """
        Check if Ollama server is running.
        
        Returns:
            bool: True if Ollama is running, False otherwise
        """
        try:
            ollama.list()
            return True
        except Exception as e:
            print(f"⚠️  Ollama server not responding: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """
        List all available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            models = ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def check_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if model is available
        """
        available_models = self.list_models()
        # Check if model exists (handle version tags)
        return any(model_name in model for model in available_models)
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama registry.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            bool: True if successful
        """
        print(f"\n📥 Pulling model: {model_name}")
        print("   This may take a few minutes...")
        try:
            ollama.pull(model_name)
            print(f"✅ Model {model_name} pulled successfully")
            return True
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            return False
    
    def verify_required_models(self, auto_pull: bool = True) -> bool:
        """
        Verify all required models are available.
        
        Args:
            auto_pull: Automatically pull missing models
            
        Returns:
            bool: True if all models are available
        """
        print("\n🔍 Verifying required models...")
        
        all_available = True
        
        for purpose, model_name in REQUIRED_MODELS.items():
            print(f"\n   Checking {purpose} model: {model_name}")
            
            if self.check_model_available(model_name):
                print(f"   ✅ {model_name} is available")
            else:
                print(f"   ⚠️  {model_name} not found")
                
                if auto_pull and OLLAMA_CONFIG['auto_pull_models']:
                    if self.pull_model(model_name):
                        print(f"   ✅ {model_name} is now available")
                    else:
                        all_available = False
                else:
                    all_available = False
                    print(f"   ❌ Please run: ollama pull {model_name}")
        
        return all_available
    
    def test_chat_model(self, test_prompt: str = "Say 'OK' if you're working.") -> bool:
        """
        Test the chat model with a simple query.
        
        Args:
            test_prompt: Prompt to test with
            
        Returns:
            bool: True if model responds correctly
        """
        print(f"\n🧪 Testing chat model: {self.chat_model}")
        try:
            response = ollama.chat(
                model=self.chat_model,
                messages=[{'role': 'user', 'content': test_prompt}],
                options={
                    'temperature': CHAT_MODEL_CONFIG['temperature'],
                    'top_p': CHAT_MODEL_CONFIG['top_p'],
                    'top_k': CHAT_MODEL_CONFIG['top_k'],
                }
            )
            answer = response['message']['content']
            print(f"   ✅ Response: {answer[:100]}")
            return True
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def test_embedding_model(self, test_text: str = "Test embedding") -> bool:
        """
        Test the embedding model.
        
        Args:
            test_text: Text to generate embedding for
            
        Returns:
            bool: True if embeddings are generated
        """
        print(f"\n🧪 Testing embedding model: {self.embedding_model}")
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=test_text
            )
            embedding = response['embedding']
            print(f"   ✅ Generated embedding: {len(embedding)} dimensions")
            print(f"   First 5 values: {embedding[:5]}")
            return True
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with model info or None
        """
        try:
            info = ollama.show(model_name)
            return info
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return None
    
    def set_preset(self, preset_name: str):
        """
        Apply a configuration preset.
        
        Args:
            preset_name: Name of preset (conservative, balanced, creative, precise)
        """
        if preset_name not in PRESETS:
            print(f"❌ Unknown preset: {preset_name}")
            print(f"   Available: {list(PRESETS.keys())}")
            return False
        
        preset = PRESETS[preset_name]
        print(f"\n⚙️  Applying preset: {preset_name}")
        print(f"   Description: {preset['description']}")
        
        for key, value in preset.items():
            if key != 'description' and key in CHAT_MODEL_CONFIG:
                CHAT_MODEL_CONFIG[key] = value
                print(f"   {key}: {value}")
        
        print("✅ Preset applied")
        return True
    
    def initialize_system(self) -> bool:
        """
        Complete system initialization and verification.
        
        Returns:
            bool: True if system is ready
        """
        print("="*80)
        print("🚀 INITIALIZING OLLAMA MODEL SYSTEM")
        print("="*80)
        
        # Check if Ollama is running
        if not self.check_ollama_running():
            print("\n❌ Ollama server is not running!")
            print("\n💡 To start Ollama:")
            print("   1. Open a new terminal")
            print("   2. Run: ollama serve")
            print("   3. Or start Ollama application")
            return False
        
        print("✅ Ollama server is running")
        
        # Verify models
        if not self.verify_required_models():
            print("\n❌ Not all required models are available")
            return False
        
        # Test models
        if not self.test_chat_model():
            print("\n❌ Chat model test failed")
            return False
        
        if not self.test_embedding_model():
            print("\n❌ Embedding model test failed")
            return False
        
        print("\n" + "="*80)
        print("✅ SYSTEM READY - All models initialized and verified")
        print("="*80)
        
        return True
    
    def print_status(self):
        """Print current system status."""
        print("\n" + "="*80)
        print("📊 OLLAMA SYSTEM STATUS")
        print("="*80)
        
        # Server status
        print("\n🔧 Server:")
        print(f"   Host: {self.host}")
        print(f"   Running: {'✅ Yes' if self.check_ollama_running() else '❌ No'}")
        
        # Chat model
        print(f"\n🤖 Chat Model:")
        print(f"   Model: {self.chat_model}")
        print(f"   Available: {'✅ Yes' if self.check_model_available(self.chat_model) else '❌ No'}")
        print(f"   Temperature: {CHAT_MODEL_CONFIG['temperature']}")
        print(f"   Top-p: {CHAT_MODEL_CONFIG['top_p']}")
        print(f"   Top-k: {CHAT_MODEL_CONFIG['top_k']}")
        
        # Embedding model
        print(f"\n🧠 Embedding Model:")
        print(f"   Model: {self.embedding_model}")
        print(f"   Available: {'✅ Yes' if self.check_model_available(self.embedding_model) else '❌ No'}")
        print(f"   Dimensions: {EMBEDDING_MODEL_CONFIG['dimensions']}")
        
        # Available models
        print(f"\n📦 Available Models:")
        models = self.list_models()
        for model in models:
            print(f"   - {model}")
        
        print("="*80)


def main():
    """Main function to demonstrate usage."""
    manager = OllamaManager()
    
    # Initialize and verify system
    if manager.initialize_system():
        # Print status
        manager.print_status()
        
        # Show how to use presets
        print("\n💡 Available presets:")
        for name, preset in PRESETS.items():
            print(f"   - {name}: {preset['description']}")
        
        print("\n📝 To use a preset:")
        print("   manager.set_preset('conservative')")
        print("   manager.set_preset('creative')")
    else:
        print("\n❌ System initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
