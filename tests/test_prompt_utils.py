import pytest

from shwizard.utils.prompt_utils import parse_command_response


def test_parse_command_response_with_separators():
    resp = "du -a . ||| ls -la"
    cmds = parse_command_response(resp)
    assert cmds == ["du -a .", "ls -la"]


def test_parse_command_response_with_code_block():
    resp = """Here are the commands:

```bash
du -a .
```
"""
    cmds = parse_command_response(resp)
    assert cmds == ["du -a ."]


def test_parse_command_response_ignores_explanations_and_bullets_and_handles_labels():
    resp = """shwizard> 今のFOlderのサイズ

🔍 処理: 今のFOlderのサイズ

✅ コマンド 1: du -a .

✅ コマンド 2: **Explanation:**

✅ コマンド 3: * `du`: This is the command to display disk usage.

✅ コマンド 4: * `-a`: This option tells `du` to list all files, including hidden ones, and directories.

✅ コマンド 5: * `.`:  This represents the current directory (".") where you want to get the sizes of all files and folders in this directory.
"""
    cmds = parse_command_response(resp)
    assert cmds == ["du -a ."]


def test_parse_command_response_english_label_with_noise():
    resp = """Command 1: ls -la
Command 2: Explanation:
- `ls`: list directory contents
"""
    cmds = parse_command_response(resp)
    assert cmds == ["ls -la"]
