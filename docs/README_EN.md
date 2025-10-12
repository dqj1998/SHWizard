# SHWizard

<div align="center">

🧙‍♂️ **An Intelligent Shell Command Assistant - Control Your Terminal with Natural Language**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

English | [简体中文](../README.md)

</div>

## 📖 Introduction

SHWizard is a cross-platform AI-powered shell assistant that lets you describe what you want to do in natural language, and it automatically generates and executes the corresponding shell commands.

### ✨ Core Features

- 🤖 **Local AI Powered** - Embedded Ollama with gemma2:2b as default model
- 🛡️ **Smart Safety Protection** - Multi-level dangerous command detection system
- 📚 **Smart History Priority** - Optimized suggestions based on command history
- 🎯 **Context Aware** - Automatically detects OS, current directory, and environment
- 💾 **Local Storage** - All data stored locally (SQLite + YAML)
- 🚀 **One-Click Installation** - Pre-built binaries ready to use
- 🌍 **Cross-Platform** - Supports Linux, macOS, and Windows

## 🚀 Quick Start

### Option 1: Use Pre-built Binary (Recommended)

```bash
# Download the binary for your platform
# Linux/macOS
curl -L https://github.com/dqj1998/SHWizard/releases/latest/download/shwizard -o shwizard
chmod +x shwizard
sudo mv shwizard /usr/local/bin/

# First run will automatically download Ollama and AI model
shwizard "list all Python files"
```

### Option 2: Install via pip

```bash
pip install shwizard

# Or install from source
git clone https://github.com/dqj1998/SHWizard.git
cd SHWizard
pip install -e .
```

## 💡 Usage Examples

### Single Query Mode

```bash
# Basic usage
shwizard "find the 10 largest files"
shwizard "show all git repositories in current directory"
shwizard "compress all jpg images"

# Explain a command
shwizard --explain "find . -name '*.py' -type f"

# View history
shwizard --history
```

### Interactive Mode

```bash
# Start interactive session
shwizard --interactive

# Or use short option
shwizard -i
```

In interactive mode:
```
shwizard> list all running Docker containers
shwizard> find code files containing TODO
shwizard> /history  # View history
shwizard> /stats    # View statistics
shwizard> /quit     # Exit
```

### Advanced Options

```bash
# Dry-run mode (show commands without executing)
shwizard --dry-run "delete all .tmp files"

# Disable safety checks (not recommended)
shwizard --no-safety "rm -rf temp/"

# Use custom config
shwizard --config-path ~/my-config.yaml "list files"
```

## 🛡️ Safety Features

### Three-Level Dangerous Command Protection

SHWizard has built-in intelligent dangerous command detection:

#### 🚨 High Risk (Blocked or requires confirmation)
- `rm -rf /` - Delete root directory
- `dd` direct disk write
- Fork bombs
- Format filesystem

#### ⚠️ Medium Risk (Warning)
- `rm -rf` - Recursive force delete
- `chmod 777` - Open all permissions
- `curl | sh` - Download and execute script
- Modify system config files

#### ℹ️ Low Risk (Info only)
- Normal delete commands
- sudo operations

### Confirmation Flow Example

```
User input: "delete all temp files"
↓
AI generates: "rm -rf /tmp/*"
↓
Safety check: [MEDIUM RISK] - Recursive delete
↓
Display:
  ⚠️  Warning: This command will recursively delete files
  Command: rm -rf /tmp/*
  Impact: Will delete all files in /tmp directory
  Continue? [y/N]
```

## ⚙️ Configuration

Config file location: `~/.config/shwizard/config.yaml`

```yaml
ollama:
  embedded: true              # Use embedded Ollama
  auto_download: true         # Auto-download models
  model: "gemma2:2b"         # Default model
  base_url: "http://localhost:11434"

safety:
  enabled: true               # Enable safety checks
  confirm_high_risk: true     # Confirm high-risk commands
  warn_medium_risk: true      # Warn medium-risk commands

history:
  enabled: true               # Enable history
  max_entries: 10000          # Max history entries
  priority_search: true       # Enable priority search

ui:
  color_enabled: true         # Enable colors
  show_explanations: true     # Show explanations
  confirm_execution: true     # Confirm before execution
```

### Modify Configuration

```bash
# View config
shwizard config ollama.model

# Change config
shwizard config ollama.model llama2

# Switch to another model
shwizard config ollama.model codellama
```

## 📊 How It Works

```
┌─────────────────────────────────────────┐
│     User inputs natural language query   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Collect context (OS, dir, tools, etc) │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Search relevant history (prioritized)  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Local AI generates shell command      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Safety check (dangerous cmd detection)│
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   User confirms and executes command    │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Save history and feedback             │
└─────────────────────────────────────────┘
```

## 🔧 Development

### Build from Source

```bash
# Clone repository
git clone https://github.com/dqj1998/SHWizard.git
cd SHWizard

# Install dev dependencies
make dev

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Build binary
make binary
```

## 📝 FAQ

### 1. First startup is slow?
First startup downloads Ollama and AI model (~1.6GB). Please be patient.

### 2. How to change AI model?
```bash
shwizard config ollama.model llama2
```

### 3. Where is history stored?
Default location: `~/.local/share/shwizard/history.db`

### 4. How to customize dangerous command rules?
Edit `~/.shwizard/custom_rules.yaml`

### 5. Which shells are supported?
Supports bash, zsh, fish, powershell, and other mainstream shells

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

[MIT License](../LICENSE)

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Rich](https://github.com/Textualize/rich) - Terminal beautification
- [Click](https://click.palletsprojects.com/) - CLI framework

---

<div align="center">

Made with ❤️ by [dqj1998](https://github.com/dqj1998)

⭐ If you find this useful, please give it a Star!

</div>
