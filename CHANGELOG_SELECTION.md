# 命令选择交互方式改进

## 修改日期

2025-10-26

## 修改内容

### 问题描述

之前用户选择推荐命令的数字后会立即执行,这样可能有风险且不方便用户修改命令。

### 解决方案

改进了命令选择的交互方式，支持单键快速选择和命令预填充：

1. **普通数字输入（按 1, 2, 3...，无需回车）**：
   - 选择命令后将其填充到 `shwizard>` 提示符后
   - 用户可以直接按 Enter 执行，或编辑后再执行
   - 按 Ctrl+C 取消

2. **感叹号+数字输入（按 `!` 然后按数字，如 `!1`）**：
   - 直接执行对应的命令（跳过编辑步骤）
   - 类似 shell 的 `!123` 历史命令执行方式
   - 无需按回车，两个按键即可完成

### 修改的文件

1. **shwizard/cli.py**
   - `process_query()` 函数中的历史命令选择逻辑
   - `select_command()` 函数中的 AI 生成命令选择逻辑

2. **shwizard/utils/i18n.py**
   - 添加新的提示文本：
     - `selection_hint`：选择提示
     - `selected_command`：已选择命令
     - `edit_or_execute_prompt`：编辑或执行提示
     - `modify_command_prompt`：修改命令提示

### 使用示例

#### 场景1：查看并编辑命令

```text
🔍 Processing: 查找所有 Python 文件

✅ Found 3 matching command(s) from history:

✅ Command 1: find . -name "*.py"
   Keywords matched: 2 | Used: 5 times

✅ Command 2: ls -la *.py
   Keywords matched: 1 | Used: 2 times

Select command [1-2] or 'q' to quit
Tip: Enter number to review/edit, or !number (e.g., !1) to execute directly
> 1                           # 只需按 1，无需回车

Selected command: find . -name "*.py"

shwizard> find . -name "*.py" -type f  # 命令自动填充，可以编辑后按回车执行
```

#### 场景2：直接执行命令

```text
Select command [1-3] or 'q' to quit
Tip: Enter number to review/edit, or !number (e.g., !1) to execute directly
> !1                          # 按 ! 然后按 1，直接执行（无需回车）

[直接执行命令 1...]
```

### 优势

1. **更安全**：默认会先显示命令内容，让用户确认
2. **更灵活**：
   - 命令自动填充到输入区，可以任意编辑
   - 使用 prompt_toolkit 的所有编辑功能（光标移动、复制粘贴等）
   - 按 Enter 直接执行，无需额外确认
3. **保持高效**：
   - 单键选择，无需按回车（按 `1` 即可，不是 `1` + `Enter`）
   - 对于熟悉的命令，可以用 `!` + 数字快速执行（按 `!` 再按 `1`）
4. **符合习惯**：
   - `!` 前缀与 shell 历史命令执行方式一致
   - 命令预填充类似 shell 的 `Ctrl+R` 搜索历史后的体验

### 向后兼容性

- 所有原有的安全检查（危险命令确认等）仍然有效
- 命令历史记录功能不受影响
- 高危命令仍会要求额外确认（输入 "yes"）

