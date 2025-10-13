import re
from typing import List

# Common commands that are often used with flags/paths/operators
COMMON_COMMANDS = {
    "ls", "pwd", "whoami", "date", "cat", "head", "tail", "tee", "echo", "ln",
    "rm", "mv", "cp", "chmod", "chown", "find", "grep", "awk", "sed", "tar", "zip", "unzip",
    "du", "df", "ps", "kill", "xargs", "tr", "sort", "uniq",
    "curl", "wget", "ssh", "scp", "rsync",
    "docker", "kubectl", "aws", "az",
    "git", "npm", "pnpm", "yarn", "node", "python", "pip", "pip3", "brew", "go", "make",
    "psql", "sqlite3", "systemctl", "service"
}

# Commands that commonly have a subcommand pattern: "<cmd> <subcmd> ..."
SUBCOMMAND_COMMANDS = {
    "git", "npm", "pnpm", "yarn", "kubectl", "docker", "aws", "az", "brew", "go", "make",
    "psql", "sqlite3", "ssh", "scp", "rsync", "python", "pip", "pip3", "node"
}

# Simple commands that are valid even without flags/paths/subcommands
SIMPLE_COMMANDS = {"ls", "pwd", "whoami", "date", "cd"}


def _contains_shell_operators(text: str) -> bool:
    # Detect typical shell syntax that strongly indicates a command
    # Note: Avoid treating backticks or stray parentheses as operators to reduce false positives from explanations
    operators = ["|", "&&", "||", ";", ">", ">>", "<", "$("]
    return any(op in text for op in operators)


def _contains_flags(text: str) -> bool:
    # Look for short or long options anywhere beyond the first token
    return bool(re.search(r"\s-\w|\s--[A-Za-z][A-Za-z0-9\-]*", text))


def _contains_path_like(text: str) -> bool:
    # Detect paths or globs typical in commands
    return bool(
        re.search(r"(\./|\.\./|/[^ \n]|~[/ ])", text) or
        re.search(r"[A-Za-z0-9_\-\/\.]+\.(sh|py|js|ts|go|c|cpp|json|yaml|yml|txt)\b", text) or
        ("*" in text or "?" in text or "[" in text or "]" in text)
    )


def is_command_input(text: str) -> bool:
    """
    Heuristic detection of whether the input is a shell command vs natural language.

    Rules:
    - If the input contains shell operators, flags, or path-like tokens -> command
    - If it starts with '$ ' or 'sudo ' -> command
    - If it starts with a known command:
      - If in SIMPLE_COMMANDS -> command
      - If in SUBCOMMAND_COMMANDS and has more than one token -> command
      - Otherwise require flags/paths/operators to avoid false positives like "find all python files"
    """
    if not text:
        return False

    s = text.strip()

    # Ignore markdown bullets, numbered lists, and explanation headings often present in LLM outputs
    if re.match(r"^\s*(?:\*|-|•|\d+\.)\s", s):
        return False
    if s.lower().startswith(("explanation", "**explanation**")):
        return False
    if s.startswith("```"):
        return False

    # Common shells prefix or sudo indicates a command
    if s.startswith("$ ") or s.startswith("sudo "):
        return True

    # Multiline input: treat any line with operator/flags/path as command
    if "\n" in s:
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            if _contains_shell_operators(line) or _contains_flags(line) or _contains_path_like(line):
                return True
        # If none of the lines look like commands, consider natural language
        return False

    # Strong indicators
    if _contains_shell_operators(s) or _contains_flags(s) or _contains_path_like(s):
        return True

    # Extract first token as potential command name
    m = re.match(r"^\s*(?:\$ )?([A-Za-z0-9][\w\.\-]*)\b", s)
    if not m:
        return False

    cmd = m.group(1)

    # If it's a simple command (like `ls`) even with no args, consider it a command
    if cmd in SIMPLE_COMMANDS:
        return True

    # If it's a subcommand-style CLI and has at least one more token, consider it a command
    tokens: List[str] = s.split()
    if cmd in SUBCOMMAND_COMMANDS and len(tokens) >= 2:
        return True

    # If it's a common command, but without strong indicators, be conservative to avoid false positives
    if cmd in COMMON_COMMANDS:
        # Require at least flags/paths/operators to treat as a command
        return _contains_flags(s) or _contains_path_like(s) or _contains_shell_operators(s)

    # Default: not a command
    return False
