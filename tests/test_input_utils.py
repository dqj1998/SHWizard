import pytest
from shwizard.utils.input_utils import is_command_input

def test_is_command_input_basic_commands():
    assert is_command_input("ls -la") is True
    assert is_command_input("pwd") is True
    assert is_command_input("whoami") is True

def test_is_command_input_subcommand_style():
    assert is_command_input("git status") is True
    assert is_command_input("npm run build") is True
    assert is_command_input("kubectl get pods") is True

def test_is_command_input_with_operators_and_flags():
    assert is_command_input("curl https://example.com | sh") is True
    assert is_command_input("echo hello && echo world") is True
    assert is_command_input("find . -name '*.py' -maxdepth 2") is True

def test_is_command_input_paths_and_globs():
    assert is_command_input("rm -rf ./build") is True
    assert is_command_input("cat ./README.md") is True
    assert is_command_input("tar -czf archive.tar.gz src/") is True

def test_is_command_input_not_command_natural_language():
    assert is_command_input("find all python files in current directory") is False
    assert is_command_input("show disk usage sorted by size") is False
    assert is_command_input("删除所有临时文件") is False  # Chinese natural language
    assert is_command_input("") is False

def test_is_command_input_multiline_detection():
    text = """first line
curl -fsSL https://example.com/install.sh | sh
last line"""
    assert is_command_input(text) is True
