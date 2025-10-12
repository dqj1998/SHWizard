# SHWizard

<div align="center">

🧙‍♂️ **一个智能的 Shell 命令助手 - 用自然语言操作命令行**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[English](./docs/README_EN.md) | 简体中文

</div>

## 📖 简介

SHWizard 是一个跨平台的 AI 辅助 Shell 工具,让你可以用自然语言描述你想做的事情,然后自动生成并执行对应的 Shell 命令。

### ✨ 核心特性

- 🤖 **本地 AI 驱动** - 使用内嵌的 Ollama,默认搭载 gemma2:2b 模型
- 🛡️ **智能安全防护** - 多级危险命令检测和拦截系统
- 📚 **智能历史优先** - 基于历史命令优化建议
- 🎯 **上下文感知** - 自动识别操作系统、当前目录等环境信息
- 💾 **本地存储** - 所有数据保存在本地（SQLite + YAML）
- 🚀 **一键安装** - 提供预编译二进制文件,开箱即用
- 🌍 **跨平台支持** - Linux、macOS、Windows 全平台支持

## 🚀 快速开始

### 方式一：使用预编译二进制（推荐）

```bash
# 下载对应平台的二进制文件
# Linux/macOS
curl -L https://github.com/dqj1998/SHWizard/releases/latest/download/shwizard -o shwizard
chmod +x shwizard
sudo mv shwizard /usr/local/bin/

# 首次运行会自动下载 Ollama 和 AI 模型
shwizard "列出所有 Python 文件"
```

### 方式二：使用 pip 安装

```bash
pip install shwizard

# 或者从源码安装
git clone https://github.com/dqj1998/SHWizard.git
cd SHWizard
pip install -e .
```

## 💡 使用示例

### 单次查询模式

```bash
# 基础用法
shwizard "找出占用空间最大的 10 个文件"
shwizard "显示当前目录下所有的 git 仓库"
shwizard "压缩所有 jpg 图片"

# 解释命令
shwizard --explain "find . -name '*.py' -type f"

# 查看历史
shwizard --history
```

### 交互模式

```bash
# 启动交互式会话
shwizard --interactive

# 或者使用短选项
shwizard -i
```

在交互模式中：
```
shwizard> 列出所有正在运行的 Docker 容器
shwizard> 查找包含 TODO 的代码文件
shwizard> /history  # 查看历史
shwizard> /stats    # 查看统计
shwizard> /quit     # 退出
```

### 高级选项

```bash
# 干运行模式（只显示命令,不执行）
shwizard --dry-run "删除所有 .tmp 文件"

# 禁用安全检查（不推荐）
shwizard --no-safety "rm -rf temp/"

# 使用自定义配置
shwizard --config-path ~/my-config.yaml "列出文件"
```

## 🛡️ 安全特性

### 三级危险命令防护

SHWizard 内置了智能的危险命令检测系统：

#### 🚨 高危命令（会被拦截或需要确认）
- `rm -rf /` - 删除根目录
- `dd` 直接写入磁盘
- Fork 炸弹
- 格式化文件系统

#### ⚠️ 中危命令（会警告）
- `rm -rf` - 递归强制删除
- `chmod 777` - 开放所有权限
- `curl | sh` - 下载并执行脚本
- 修改系统配置文件

#### ℹ️ 低危命令（仅提示）
- 普通删除命令
- sudo 操作

### 确认流程示例

```
用户输入: "删除所有临时文件"
↓
AI 生成: "rm -rf /tmp/*"
↓
安全检查: [中危] - 递归删除
↓
显示:
  ⚠️  警告: 此命令将递归删除文件
  命令: rm -rf /tmp/*
  影响: 将删除 /tmp 目录下所有文件
  是否继续? [y/N]
```

## ⚙️ 配置

配置文件位置：`~/.config/shwizard/config.yaml`

```yaml
ollama:
  embedded: true              # 使用内嵌 Ollama
  auto_download: true         # 自动下载模型
  model: "gemma2:2b"         # 默认模型
  base_url: "http://localhost:11434"

safety:
  enabled: true               # 启用安全检查
  confirm_high_risk: true     # 高危命令需确认
  warn_medium_risk: true      # 中危命令警告

history:
  enabled: true               # 启用历史记录
  max_entries: 10000          # 最大历史条目
  priority_search: true       # 启用优先搜索

ui:
  color_enabled: true         # 启用彩色输出
  show_explanations: true     # 显示解释
  confirm_execution: true     # 执行前确认
```

### 修改配置

```bash
# 查看配置
shwizard config ollama.model

# 修改配置
shwizard config ollama.model llama2

# 切换到其他模型
shwizard config ollama.model codellama
```

## 📊 工作原理

```
┌─────────────────────────────────────────┐
│     用户输入自然语言查询                 │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   收集上下文信息（OS、目录、工具等）      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   搜索相关历史命令（优先级排序）          │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   本地 AI 生成 Shell 命令                │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   安全检查（危险命令检测）                │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   用户确认并执行命令                     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   保存历史和反馈                         │
└─────────────────────────────────────────┘
```

## 🔧 开发

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/dqj1998/SHWizard.git
cd SHWizard

# 安装开发依赖
make dev

# 运行测试
make test

# 代码格式化
make format

# 代码检查
make lint

# 构建二进制文件
make binary
```

### 项目结构

```
SHWizard/
├── shwizard/
│   ├── core/              # 核心功能
│   │   ├── ai_service.py  # AI 服务
│   │   ├── context_manager.py
│   │   └── executor.py
│   ├── safety/            # 安全检查
│   │   ├── checker.py
│   │   └── rules.py
│   ├── storage/           # 数据存储
│   │   ├── database.py
│   │   ├── config.py
│   │   └── history.py
│   ├── llm/              # LLM 管理
│   │   └── ollama_manager.py
│   ├── utils/            # 工具函数
│   └── data/             # 配置和规则
├── tests/                # 测试
└── docs/                 # 文档
```

## 📝 常见问题

### 1. 首次启动很慢？
首次启动会自动下载 Ollama 和 AI 模型（约 1.6GB）,请耐心等待。

### 2. 如何更换 AI 模型？
```bash
shwizard config ollama.model llama2
```

### 3. 历史记录存在哪里？
默认存储在 `~/.local/share/shwizard/history.db`

### 4. 如何自定义危险命令规则？
编辑 `~/.shwizard/custom_rules.yaml`

### 5. 支持哪些 Shell？
支持 bash、zsh、fish、powershell 等主流 Shell

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[MIT License](LICENSE)

## 🙏 致谢

- [Ollama](https://ollama.ai/) - 本地 LLM 运行环境
- [Rich](https://github.com/Textualize/rich) - 终端美化
- [Click](https://click.palletsprojects.com/) - CLI 框架

---

<div align="center">

Made with ❤️ by [dqj1998](https://github.com/dqj1998)

⭐ 如果觉得有用,请给个 Star！

</div>
