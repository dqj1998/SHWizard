# SHWizard

<div align="center">

🧙‍♂️ An intelligent Shell command assistant — operate the command line with natural language

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


</div>

## 📖 Introduction

SHWizard is a cross-platform AI-assisted Shell tool that lets you describe what you want to do in natural language, then automatically generates and executes the corresponding Shell commands.

### ✨ Core Features

- 🤖 Local AI powered — uses embedded LLM_cpp with GGUF models, ships with Gemma 2B by default
- 🛡️ Intelligent safety protection — multi-level dangerous command detection and interception
- 📚 History-aware prioritization — suggestions optimized using past command history
- 🎯 Context awareness — automatically recognizes OS, current directory, and environment details
- 💾 Local storage — all data is stored locally (SQLite + YAML)
- 🚀 One-click install — precompiled binaries, ready to use out of the box
- 🌍 Cross-platform support — works on Linux, macOS, and Windows

## 🚀 Quick Start

### Method 1: Use precompiled binary (recommended)

```bash
# Download the binary for your platform
# Linux/macOS
curl -L https://github.com/dqj1998/SHWizard/releases/latest/download/shwizard -o shwizard
chmod +x shwizard
sudo mv shwizard /usr/local/bin/

# On first run, the AI model will be downloaded automatically
shwizard "List all Python files"
```

### Method 2: Install via pip

```bash
pip install shwizard

# Or install from source
git clone https://github.com/dqj1998/SHWizard.git
cd SHWizard
pip install -e .
```

## 💡 Usage Examples

### Single-shot mode

```bash
# Basic usage
shwizard "Find the top 10 largest files by size"
shwizard "Show all git repositories in the current directory"
shwizard "Compress all jpg images"

# Explain a command
shwizard --explain "find . -name '*.py' -type f"

# View history
shwizard --history
```

### Interactive mode

```bash
# Start an interactive session
shwizard --interactive

# Or use the short option
shwizard -i
```

In interactive mode:
```
shwizard> List all running Docker containers
shwizard> Find code files containing TODO
shwizard> /history  # View history
shwizard> /models   # List available models
shwizard> /switch gemma-2-2b-it-q4_k_m.gguf  # Switch model
shwizard> /info     # Show model information
shwizard> /stats    # View statistics
shwizard> /quit     # Exit
```

### Advanced options

```bash
# Dry-run mode (show the command without executing)
shwizard --dry-run "Delete all .tmp files"

# Disable safety checks (not recommended)
shwizard --no-safety "rm -rf temp/"

# Use a custom configuration
shwizard --config-path ~/my-config.yaml "List files"
```

## 🛡️ Security Features

### Three-level dangerous command protection

SHWizard includes an intelligent dangerous command detection system:

#### 🚨 High-risk commands (will be blocked or require confirmation)
- `rm -rf /` — delete root directory
- `dd` writing directly to disk
- Fork bombs
- Formatting file systems

#### ⚠️ Medium-risk commands (will warn)
- `rm -rf` — recursive forced deletion
- `chmod 777` — grant full permissions to everyone
- `curl | sh` — download and execute scripts
- Modifying system configuration files

#### ℹ️ Low-risk commands (informational prompt)
- Regular delete commands
- sudo operations

### Confirmation flow example

```
User input: "Delete all temporary files"
↓
AI generates: "rm -rf /tmp/*"
↓
Safety check: [Medium] — Recursive deletion
↓
Display:
  ⚠️  Warning: This command will recursively delete files
  Command: rm -rf /tmp/*
  Impact: Will delete all files under the /tmp directory
  Continue? [y/N]
```

## ⚙️ Configuration

Configuration file location: `~/.config/shwizard/config.yaml`

```yaml
llm:
  backend: "llmcpp"           # Use LLM_cpp backend
  model: "gemma-2-2b-it-q4_k_m.gguf"  # Default GGUF model
  # Model storage directory (platform-specific):
  # macOS: ~/Library/Application Support/shwizard/models/
  # Linux: ~/.local/share/shwizard/models/
  # Windows: %LOCALAPPDATA%/shwizard/models/
  auto_download: true         # Automatically download models
  n_ctx: 2048                 # Context window size
  n_gpu_layers: -1            # GPU layers (-1 for auto-detect)
  temperature: 0.7            # Generation temperature

safety:
  enabled: true               # Enable safety checks
  confirm_high_risk: true     # Require confirmation for high-risk commands
  warn_medium_risk: true      # Warn on medium-risk commands

history:
  enabled: true               # Enable history
  max_entries: 10000          # Maximum history entries
  priority_search: true       # Enable priority search

ui:
  color_enabled: true         # Enable colored output
  show_explanations: true     # Show explanations
  confirm_execution: true     # Confirm before execution
```

### Modify configuration

```bash
# View current model information
shwizard -i
shwizard> /info

# List available models
shwizard> /models

# Switch to a different model
shwizard> /switch llama-3.2-3b-instruct-q4_k_m.gguf

# Or modify configuration file directly
# Edit ~/.shwizard/config.yaml
```

## 📊 How It Works

```
┌─────────────────────────────────────────┐
│     User enters a natural language query │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Collect context (OS, directory, tools) │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Search related history (priority sorted) │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Local LLM_cpp generates Shell commands │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Safety checks (dangerous command detection) │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   User confirms and executes commands   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Save history and feedback             │
└─────────────────────────────────────────┘
```

## 🔧 Development

### Build from source

```bash
# Clone the repository
git clone https://github.com/dqj1998/SHWizard.git
cd SHWizard

# Install development dependencies
make dev

# Run tests
make test

# Code formatting
make format

# Linting
make lint

# Build the binary
make binary
```

### Project structure

```
SHWizard/
├── shwizard/
│   ├── core/              # Core functionality
│   │   ├── ai_service.py  # AI service
│   │   ├── context_manager.py
│   │   └── executor.py
│   ├── safety/            # Safety checks
│   │   ├── checker.py
│   │   └── rules.py
│   ├── storage/           # Data storage
│   │   ├── database.py
│   │   ├── config.py
│   │   └── history.py
│   ├── llm/               # LLM management
│   │   ├── llmcpp_manager.py
│   │   ├── model_downloader.py
│   │   └── model_registry.py
│   ├── utils/             # Utility functions
│   └── data/              # Configurations and rules
├── tests/                 # Tests
└── docs/                  # Documentation
```

## 📝 FAQ

### 1. Is the first startup slow?
On first launch, the AI model (~1.5GB) will be downloaded automatically. Please be patient.

### 2. How do I switch the AI model?
```bash
# In interactive mode
shwizard -i
shwizard> /models          # List available models
shwizard> /switch <model>  # Switch to a specific model
```

### 3. Where is the history stored?
By default at `~/.local/share/shwizard/history.db`

### 4. How do I customize dangerous command rules?
Edit `~/.shwizard/custom_rules.yaml`

### 5. Which shells are supported?
Supports major shells including bash, zsh, fish, and powershell.

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — efficient LLM inference
- [Rich](https://github.com/Textualize/rich) — terminal styling
- [Click](https://click.palletsprojects.com/) — CLI framework

---

<div align="center">

Made with ❤️ by [dqj1998](https://github.com/dqj1998)

⭐ If you find it useful, please give it a Star!

</div>
