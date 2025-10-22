from typing import Dict, List, Optional
import re
from shwizard.utils.input_utils import is_command_input


SYSTEM_PROMPT_TEMPLATE = """The command to {user_query} is: """


EXPLAIN_PROMPT_TEMPLATE = """Explain this shell command in simple terms:

Command: {command}

Operating System: {os_type}
Shell Type: {shell_type}

Provide a clear, concise explanation of:
1. What the command does
2. Each part of the command
3. Potential risks or side effects
4. Expected output

Explanation:"""


def build_command_prompt(
    user_query: str,
    os_type: str,
    cwd: str,
    shell_type: str,
    installed_tools: List[str],
    history_commands: Optional[List[str]] = None
) -> str:
    tools_str = ", ".join(installed_tools[:10]) if installed_tools else "standard tools"
    
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        os_type=os_type,
        cwd=cwd,
        shell_type=shell_type,
        installed_tools=tools_str,
        user_query=user_query
    )
    
    if history_commands:
        history_str = "\n".join([f"- {cmd}" for cmd in history_commands[:5]])
        prompt += f"\n\nRelevant commands from history:\n{history_str}\n\nConsider these historical commands when generating new ones."
    
    return prompt


def build_explain_prompt(command: str, os_type: str, shell_type: str) -> str:
    return EXPLAIN_PROMPT_TEMPLATE.format(
        command=command,
        os_type=os_type,
        shell_type=shell_type
    )


def parse_command_response(response: str) -> List[str]:
    """
    Parse LLM response and extract only valid shell commands.

    Rules:
    - Prefer content inside fenced code blocks if present.
    - Support "|||" as multi-command separator.
    - Strip markdown bullets, numbered lists, and "Command/コマンド N:" prefixes.
    - Ignore "Explanation:" lines and code fence markers.
    - Trim leading "$ " prompt prefix.
    - Validate each candidate line using is_command_input and deduplicate.
    """
    if not response or not response.strip():
        return []
    
    text = response.strip()

    # Prefer fenced code blocks if present: extract inner content(s)
    blocks: List[str] = []
    if "```" in text:
        parts = text.split("```")
        # Code blocks are at odd indices: 1,3,5,...
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if block:
                blocks.append(block)
    if blocks:
        text = "\n".join(blocks)

    # Split into candidates using "|||", otherwise keep as single candidate
    candidates: List[str] = [c.strip() for c in (text.split("|||") if "|||" in text else [text])]

    cleaned: List[str] = []
    for cand in candidates:
        # Process line-by-line to filter explanation and markdown noise
        for line in cand.splitlines():
            l = line.strip()
            if not l:
                continue

            # Strip decorative emoji/symbol prefixes (e.g., '✅', '🔍', '👉')
            l = re.sub(r"^\s*(?:[^\w$`]+)\s*", "", l)

            # Remove common prefixes used by LLM enumerations
            l = re.sub(r"^\s*(?:[-*•]\s|\d+\.\s)", "", l)
            l = re.sub(r"^\s*(?:Command|コマンド)\s*\d+\s*:\s*", "", l, flags=re.IGNORECASE)

            # Strip tool/prompt prefixes like 'shwizard> ' or 'user>'
            l = re.sub(r"^\s*[\w\-\s]+>\s*", "", l)

            # Ignore explanation headings (robust to leading symbols and optional colon/formatting)
            if (
                re.match(r"^\s*(?:[^\w$`]+)?\s*explanation\b", l, flags=re.IGNORECASE)
                or re.match(r"^\s*\*\*explanation:? ?\*\*\s*$", l, flags=re.IGNORECASE)
            ):
                continue

            # Strip shell prompt prefix
            if l.startswith("$ "):
                l = l[2:]

            # Skip residual code fence markers
            if l.startswith("```") or l.endswith("```"):
                continue

            # Accept only lines that look like actual shell commands
            if is_command_input(l):
                cleaned.append(l)

    # Deduplicate while preserving order
    result: List[str] = []
    seen = set()
    for cmd in cleaned:
        if cmd not in seen:
            seen.add(cmd)
            result.append(cmd)

    return result
