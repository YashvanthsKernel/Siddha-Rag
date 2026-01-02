# Ollama Model Management

This folder contains all Ollama model configuration and management files.

## Files

### `config.py`
Central configuration file for all model parameters:
- Ollama server settings
- Chat model configuration (temperature, top_p, top_k, etc.)
- Embedding model configuration  
- Vector database settings
- Text processing parameters
- Preset configurations (conservative, balanced, creative, precise)

**Customize all parameters here!**

### `model_manager.py`
Ollama model manager class that handles:
- Model availability checks
- Automatic model pulling
- Health checks and testing
- Configuration presets
- System initialization

## Usage

### Initialize System
```python
from models import OllamaManager

manager = OllamaManager()
manager.initialize_system()  # Checks and verifies everything
```

### Check Status
```python
manager.print_status()  # Shows current configuration
```

### Use Presets
```python
# Apply different response styles
manager.set_preset('conservative')  # More factual
manager.set_preset('balanced')      # Default
manager.set_preset('creative')      # More varied
manager.set_preset('precise')       # Most deterministic
```

### Customize Parameters
Edit `config.py`:
```python
CHAT_MODEL_CONFIG = {
    'temperature': 0.7,      # Change this
    'top_p': 0.9,           # And this
    'model_name': 'llama3.2:3b'
}
```

## Quick Start

```bash
# Test the configuration
python models/model_manager.py

# Or view configuration
python models/config.py
```

## Configuration Options

### Temperature (0.0 - 1.0)
- `0.0-0.3`: Very focused, factual
- `0.4-0.7`: Balanced
- `0.8-1.0`: Creative, varied

### Top-k (number)
- Lower (10-20): More focused
- Medium (30-50): Balanced
- Higher (60+): More diverse

### Top-p (0.0 - 1.0)
- `0.7-0.8`: Conservative
- `0.9`: Balanced
- `0.95+`: Creative

## Presets Available

1. **conservative**: Factual, focused responses
2. **balanced**: Default, good balance
3. **creative**: More varied, creative
4. **precise**: Most deterministic

Choose based on your use case!
