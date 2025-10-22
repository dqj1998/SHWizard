# 🚀 Getting Started with SHWizard (LLM_cpp Backend)

Welcome to SHWizard with the new LLM_cpp backend! This guide will help you get up and running quickly.

## ✅ **Prerequisites**

Make sure you have Python 3.8+ installed:

```bash
python --version  # Should show 3.8 or higher
```

## 📦 **Installation**

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For GPU acceleration (optional but recommended):
# On macOS with Apple Silicon:
pip install llama-cpp-python[metal]

# On Windows/Linux with NVIDIA GPU:
pip install llama-cpp-python[cuda]

# For model downloads:
pip install huggingface_hub
```

### 2. Verify Installation

```bash
# Test basic functionality
python test_shwizard.py
```

You should see:

```
🧙‍♂️ Testing SHWizard Basic Functionality
==================================================
1️⃣ Testing imports...
   ✅ All imports successful
2️⃣ Testing hardware detection...
   ✅ GPU capabilities: False CUDA, True Metal
...
🎉 All basic functionality tests passed!
```

## 🎯 **Quick Start**

### Method 1: Interactive Mode (Recommended)

```bash
python -m shwizard --interactive
```

This will:

1. Automatically migrate your configuration from Ollama to LLM_cpp
2. Download the default model (Gemma 2B, ~1.5GB) on first run
3. Start the interactive shell

### Method 2: Single Commands

```bash
# Generate a command
python -m shwizard "find all python files"

# Explain a command
python -m shwizard --explain "ls -la"

# View history
python -m shwizard --history
```

## 🔧 **Interactive Commands**

Once in interactive mode, you can use these commands:

```bash
shwizard> /help      # Show all commands
shwizard> /models    # List available models
shwizard> /info      # Show current model info
shwizard> /switch <model>  # Switch to different model
shwizard> /history   # View command history
shwizard> /stats     # Show usage statistics
shwizard> /quit      # Exit
```

## 📋 **Example Session**

```bash
$ python -m shwizard -i

╭──────────────────────────────────────────╮
│ SHWizard Interactive Mode                │
│ Type your queries in natural language    │
│ Commands: /help, /models, /info, /quit   │
╰──────────────────────────────────────────╯

✅ Ready!

shwizard> /models
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model                      ┃ Size     ┃ Status   ┃ Description                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma-2-2b-it-q4_k_m.gguf  │ 1500MB   │ Download │ Fast and efficient 2B parameter...  │
│ qwen2-1.5b-instruct-q4...  │ 1200MB   │ Download │ Very fast 1.5B parameter model...   │
└────────────────────────────┴──────────┴──────────┴──────────────────────────────────────┘

shwizard> find all python files larger than 1MB
🔍 Processing: find all python files larger than 1MB

✅ Generated 1 command(s):

✅ Command 1: find . -name "*.py" -size +1M -type f

🚀 Execute this command? [Y/n]: y

[Output of the command...]

shwizard> /info
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property           ┃ Value                                ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Backend            │ llmcpp                               │
│ Model              │ gemma-2-2b-it-q4_k_m.gguf           │
│ File Size          │ 1500MB                               │
│ Context Size       │ 2048                                 │
│ GPU Layers         │ -1                                   │
│ Platform           │ macos                                │
└────────────────────┴──────────────────────────────────────┘
```

## ⚙️ **Configuration**

SHWizard will automatically create a configuration file at `~/.shwizard/config.yaml`:

```yaml
llm:
  backend: "llmcpp"
  model: "gemma-2-2b-it-q4_k_m.gguf"
  # Model storage directory (platform-specific):
  # macOS: ~/Library/Application Support/shwizard/models/
  # Linux: ~/.local/share/shwizard/models/
  # Windows: %LOCALAPPDATA%/shwizard/models/
  auto_download: true
  n_ctx: 2048
  n_gpu_layers: -1 # Auto-detect
  temperature: 0.7
```

## 🔧 **Troubleshooting**

### Model Download Issues

If you see download errors:

1. **Install huggingface_hub**:

   ```bash
   pip install huggingface_hub
   ```

2. **Manual model download**:

   ```bash
   # Create models directory (platform-specific)
   # macOS:
   mkdir -p ~/Library/Application\ Support/shwizard/models/
   # Linux:
   mkdir -p ~/.local/share/shwizard/models/
   # Windows (PowerShell):
   New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\shwizard\models"

   # Download a model manually (example)
   # Visit https://huggingface.co/models?library=gguf
   # Download a .gguf file to the appropriate models directory above
   ```

3. **Use a smaller model**:
   ```bash
   shwizard> /switch qwen2-1.5b-instruct-q4_k_m.gguf
   ```

### GPU Issues

If GPU acceleration isn't working:

1. **Check GPU support**:

   ```bash
   python test_shwizard.py  # Shows GPU capabilities
   ```

2. **Install GPU-specific version**:

   ```bash
   # For NVIDIA GPUs
   pip install llama-cpp-python[cuda]

   # For Apple Silicon
   pip install llama-cpp-python[metal]
   ```

3. **Force CPU mode**:
   Edit `~/.shwizard/config.yaml`:
   ```yaml
   llm:
     n_gpu_layers: 0 # Force CPU-only
   ```

### Memory Issues

If you get out-of-memory errors:

1. **Use a smaller model**:

   ```bash
   shwizard> /switch qwen2-1.5b-instruct-q4_k_m.gguf
   ```

2. **Reduce context size**:
   Edit config:
   ```yaml
   llm:
     n_ctx: 1024 # Reduce from 2048
   ```

## 🆕 **What's New in LLM_cpp Backend**

- ✅ **No server required** - Direct model integration
- ✅ **Better performance** - Optimized inference
- ✅ **GPU acceleration** - CUDA, Metal, OpenCL support
- ✅ **Model management** - Easy switching between models
- ✅ **Automatic migration** - Seamless upgrade from Ollama
- ✅ **Platform optimization** - Hardware-specific optimizations

## 🤝 **Need Help?**

1. **Check the logs**: Look for error messages in the terminal
2. **Run diagnostics**: `python test_shwizard.py`
3. **View model info**: Use `/info` command in interactive mode
4. **Check configuration**: `~/.shwizard/config.yaml`

## 🎉 **You're Ready!**

Start exploring with natural language commands:

- "show me the largest files in this directory"
- "find all git repositories"
- "compress all images in this folder"
- "show running processes using the most CPU"

Happy shell wizarding! 🧙‍♂️✨
