from typing import Dict, List, Optional


SYSTEM_PROMPT_TEMPLATE = """You are an expert shell command assistant. Users will describe what they want to do in natural language, and you need to generate the corresponding shell commands.

Current Environment:
- Operating System: {os_type}
- Current Directory: {cwd}
- Shell Type: {shell_type}
- Installed Tools: {installed_tools}

Requirements:
1. Generate only the command, no explanations or markdown
2. Ensure commands work on the user's system
3. If there are multiple approaches, provide 2-3 options separated by "|||"
4. Prioritize safety and readability
5. Consider the current directory context
6. Use common tools when possible

User Request: {user_query}

Generate the shell command(s):"""


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
    if "|||" in response:
        commands = [cmd.strip() for cmd in response.split("|||")]
    else:
        commands = [response.strip()]
    
    cleaned_commands = []
    for cmd in commands:
        cmd = cmd.strip()
        if cmd.startswith("```"):
            lines = cmd.split("\n")
            cmd = "\n".join(line for line in lines if not line.startswith("```"))
            cmd = cmd.strip()
        
        if cmd.startswith("$ "):
            cmd = cmd[2:]
        
        if cmd:
            cleaned_commands.append(cmd)
    
    return cleaned_commands
