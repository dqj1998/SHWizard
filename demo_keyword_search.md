# 关键词历史命令推荐功能演示

## 功能特点

### 1. 无需回车的快捷选择
当用户输入自然语言查询后，如果历史中有匹配的命令，系统会：
- 使用LLM提取最多6个关键词
- 在历史数据库中搜索匹配命令
- 按匹配关键词数量从多到少排序显示

用户可以：
- **直接按数字键（1-9）**：立即选择对应命令，无需按回车
- **按 'n' 键**：跳过历史推荐，让AI生成新命令
- **按 'q' 键**：取消操作

### 2. 详细的命令信息显示
每条推荐的历史命令会显示：
- 命令内容
- 匹配的关键词数量
- 历史执行次数
- 安全风险提示（如果有）

### 3. 示例

#### 用户输入：
```
find all python files
```

#### 系统响应：
```
🔍 Processing: find all python files

✅ Found 3 matching command(s) from history:

✅ Command 1: find . -name "*.py"
   Keywords matched: 3 | Used: 5 times

✅ Command 2: ls -la | grep .py
   Keywords matched: 2 | Used: 3 times

✅ Command 3: locate *.py
   Keywords matched: 2 | Used: 1 times

Select command [1-3] or 'q' to quit (or 'n' for new AI suggestion) 
```

此时用户只需按 `1`、`2`、`3`、`n` 或 `q` 即可，无需按回车！

## 配置选项

在 `~/.shwizard/config.yaml` 中：
```yaml
history:
  keyword_search: true  # 启用/禁用关键词搜索功能
```

## 技术实现

1. **ai_service.py**: `extract_keywords()` - 使用LLM提取关键词
2. **database.py**: `search_by_keywords()` - 数据库层面的关键词搜索和排序
3. **history.py**: `search_by_keywords()` - 历史管理器的包装方法
4. **cli.py**: 使用 `readchar.readkey()` 实现单键选择，无需回车
