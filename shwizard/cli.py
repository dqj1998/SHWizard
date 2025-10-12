import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from typing import Optional, List, Tuple
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from shwizard.core.ai_service import AIService
from shwizard.core.context_manager import ContextManager
from shwizard.core.executor import CommandExecutor
from shwizard.safety.checker import SafetyChecker
from shwizard.storage.config import ConfigManager
from shwizard.storage.history import HistoryManager
from shwizard.utils.logger import setup_logger, get_logger
from shwizard.utils.input_utils import is_command_input
from shwizard.utils.i18n import LLMTranslator, tr_llm, translate_rule_description_llm

console = Console()
logger = None


def init_logger(config: ConfigManager):
    global logger
    logger = setup_logger(
        level=config.get("logging.level", "INFO"),
        log_file=config.get("logging.file"),
        max_size_mb=config.get("logging.max_size_mb", 10),
        backup_count=config.get("logging.backup_count", 3)
    )
    return logger


@click.group(invoke_without_command=True)
@click.argument('query', nargs=-1, required=False)
@click.option('--interactive', '-i', is_flag=True, help='Start interactive mode')
@click.option('--explain', '-e', is_flag=True, help='Explain a command')
@click.option('--history', '-h', is_flag=True, help='View command history')
@click.option('--dry-run', '-d', is_flag=True, help='Show commands without executing')
@click.option('--no-safety', is_flag=True, help='Disable safety checks')
@click.option('--config-path', type=click.Path(), help='Custom config file path')
@click.pass_context
def main(ctx, query, interactive, explain, history, dry_run, no_safety, config_path):
    if ctx.invoked_subcommand is not None:
        return
    
    config = ConfigManager()
    global logger
    logger = init_logger(config)
    
    if history:
        show_history(config)
        return
    
    if interactive:
        interactive_mode(config, dry_run, not no_safety)
        return
    
    if not query:
        console.print("[yellow]Usage: shwizard <your question in natural language>[/yellow]")
        console.print("\nExamples:")
        console.print("  shwizard find all python files")
        console.print("  shwizard --interactive")
        console.print("  shwizard --history")
        return
    
    query_text = " ".join(query)
    
    if explain:
        explain_command(query_text, config)
    else:
        process_query(query_text, config, dry_run, not no_safety)


def process_query(query: str, config: ConfigManager, dry_run: bool = False, safety_enabled: bool = True):
    ai_service = AIService(
        model=config.get("ollama.model", "gemma3:270m"),
        base_url=config.get("ollama.base_url", "http://localhost:11434"),
        timeout=config.get("ollama.timeout", 60)
    )
    translator = LLMTranslator(ai_service)

    # Language preference and detection logic:
    # - Load preferred language (for command-like inputs)
    # - Detect language for natural language queries and persist it
    history_manager = HistoryManager()
    preferred_lang = history_manager.get_preferred_language("en")
    is_cmd = is_command_input(query)
    detected_lang = translator.detect(query)
    lang = detected_lang if not is_cmd else (preferred_lang or detected_lang)
    if not is_cmd:
        try:
            if lang and lang != preferred_lang:
                history_manager.set_preferred_language(lang)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to persist preferred language: {e}")

    console.print(f"\n🔍 [cyan]{tr_llm('processing', lang, translator)}:[/cyan] {query}\n")

    try:
        # Build context early so both direct-command and AI paths can record history with context
        context_manager = ContextManager(
            collect_tools=config.get("context.collect_installed_tools", False)
        )
        context = context_manager.get_context()
        
        safety_checker = SafetyChecker(enabled=safety_enabled)
        executor = CommandExecutor(dry_run=dry_run)

        # If the input looks like a direct shell command, bypass AI generation
        if is_command_input(query):
            # Perform safety check
            check_result = safety_checker.check_command(query)
            display_command_with_safety(query, check_result, 1, lang, translator)
            
            # Blocked commands should never execute
            if check_result.is_blocked():
                console.print(f"[red bold]⛔ {tr_llm('blocked_reason', lang, translator)}[/red bold]")
                return
            
            # For high-risk commands, require explicit confirmation
            if check_result.risk_level == "high_risk":
                if not confirm_dangerous_command(query, check_result, lang, translator):
                    console.print(f"[yellow]{tr_llm('execution_cancelled', lang, translator)}[/yellow]")
                    return
            
            # Record and execute directly (no generic execution confirmation for direct non-high-risk commands)
            command_id = history_manager.add_command(query, query, context)
            success, output = executor.execute(query)
            
            if output:
                console.print(f"\n[bold]{tr_llm('output_label', lang, translator)}:[/bold]")
                console.print(Panel(output, expand=False))
            
            history_manager.mark_executed(command_id, success, output[:500] if output else None)
            
            # Skip post-execution feedback prompts for direct commands
            return
        
        # Natural language path: use AI to generate commands
        
        with console.status("[bold green]Initializing AI service..."):
            if not ai_service.initialize():
                console.print("[red]❌ Failed to initialize AI service[/red]")
                console.print("[yellow]Please ensure Ollama is installed and running[/yellow]")
                return
        
        relevant_commands = []
        if config.get("history.priority_search", True):
            relevant_commands = history_manager.search_relevant_commands(query, limit=5, context=context)
        
        with console.status("[bold green]Generating commands..."):
            commands_raw = ai_service.generate_commands(query, context, relevant_commands)
        
        commands = normalize_commands(commands_raw)
        
        if not commands:
            console.print("[red]❌ Failed to generate commands[/red]")
            return
        
        console.print(f"[green]✅ Generated {len(commands)} command(s):[/green]\n")
        
        selected_command, selected_by_number = select_command(commands, safety_checker, config, lang, translator)
        
        if not selected_command:
            console.print("[yellow]Operation cancelled[/yellow]")
            return
        
        command_id = history_manager.add_command(query, selected_command, context)
        
        # Keep generic execution confirmation only for AI-generated commands
        if config.get("ui.confirm_execution", True) and not dry_run and not selected_by_number:
            if not Confirm.ask(f"\n🚀 {tr_llm('execute_this_command', lang, translator)}", default=True):
                console.print(f"[yellow]{tr_llm('execution_cancelled', lang, translator)}[/yellow]")
                return
        
        success, output = executor.execute(selected_command)
        
        if output:
            console.print(f"\n[bold]{tr_llm('output_label', lang, translator)}:[/bold]")
            console.print(Panel(output, expand=False))
        
        history_manager.mark_executed(command_id, success, output[:500] if output else None)
        
        if success and not dry_run:
            if Confirm.ask("\n✨ Was this helpful?", default=True):
                history_manager.add_feedback(command_id, 1)
            else:
                history_manager.add_feedback(command_id, -1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        console.print(f"[red]❌ Error: {e}[/red]")


def normalize_commands(commands: List[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for s in commands:
        if not s:
            continue
        for line in s.splitlines():
            cmd = line.strip()
            if not cmd:
                continue
            if is_command_input(cmd):
                if cmd not in seen:
                    normalized.append(cmd)
                    seen.add(cmd)
    return normalized

def select_command(commands: List[str], safety_checker: SafetyChecker, config: ConfigManager, lang: str, translator: LLMTranslator) -> Tuple[Optional[str], bool]:
    if len(commands) == 1:
        command = commands[0]
        check_result = safety_checker.check_command(command)
        display_command_with_safety(command, check_result, 1, lang, translator)
        
        if check_result.is_blocked():
            console.print(f"[red bold]⛔ {tr_llm('blocked_reason', lang, translator)}[/red bold]")
            return None, False
        
        if check_result.needs_confirmation():
            if not confirm_dangerous_command(command, check_result, lang, translator):
                return None
        
        return command, False
    
    # Display commands with numbers and safety indicators
    for idx, cmd in enumerate(commands, 1):
        check_result = safety_checker.check_command(cmd)
        display_command_with_safety(cmd, check_result, idx, lang, translator)
    
    while True:
        choice = Prompt.ask(
            "\n" + tr_llm("select_command_prompt", lang, translator, max=len(commands)),
            default="1"
        )
        
        if choice.lower() == 'q':
            return None, False
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(commands):
                selected = commands[idx]
                check_result = safety_checker.check_command(selected)
                
                if check_result.is_blocked():
                    console.print(f"[red bold]⛔ {tr_llm('blocked_reason', lang, translator)}[/red bold]")
                    continue
                
                if check_result.needs_confirmation():
                    if not confirm_dangerous_command(selected, check_result, lang, translator):
                        continue
                
                return selected, True
            else:
                console.print(f"[red]{tr_llm('enter_number_between', lang, translator, max=len(commands))}[/red]")
        except ValueError:
            console.print(f"[red]{tr_llm('invalid_input', lang, translator)}[/red]")


def display_command_with_safety(command: str, check_result, index: int = 1, lang: str = "en", translator: LLMTranslator = None):
    emoji = "✅" if check_result.is_safe else "⚠️"
    console.print(f"\n{emoji} [bold]{tr_llm('command_label', lang, translator)} {index}:[/bold] [white]{command}[/white]")
    
    if not check_result.is_safe or check_result.needs_warning():
        risk_emoji = {"high_risk": "🚨", "medium_risk": "⚠️", "low_risk": "ℹ️"}.get(
            check_result.risk_level, ""
        )
        risk_color = {"high_risk": "red", "medium_risk": "yellow", "low_risk": "blue"}.get(
            check_result.risk_level, "white"
        )
        desc = translate_rule_description_llm(check_result.message, lang, translator)
        console.print(f"{risk_emoji} [{risk_color}]{tr_llm('warning_label', lang, translator)}: {desc}[/{risk_color}]")


def confirm_dangerous_command(command: str, check_result, lang: str, translator: LLMTranslator) -> bool:
    console.print(f"\n[red bold]🚨 {tr_llm('danger_detected_title', lang, translator)}[/red bold]")
    console.print(f"[yellow]{tr_llm('command_label', lang, translator)}:[/yellow] {command}")
    console.print(f"[yellow]{tr_llm('risk_label', lang, translator)}:[/yellow] {translate_rule_description_llm(check_result.message, lang, translator)}")
    
    confirmation = Prompt.ask(
        "\n" + tr_llm("type_yes_to_proceed", lang, translator),
        default="no"
    )
    
    return confirmation.lower() == "yes"


def explain_command(command: str, config: ConfigManager):
    console.print(f"\n🔍 [cyan]Explaining:[/cyan] {command}\n")
    
    try:
        context_manager = ContextManager()
        context = context_manager.get_context()
        
        ai_service = AIService(
            model=config.get("ollama.model", "gemma3:270m"),
            base_url=config.get("ollama.base_url", "http://localhost:11434")
        )
        
        with console.status("[bold green]Initializing AI service..."):
            if not ai_service.initialize():
                console.print("[red]❌ Failed to initialize AI service[/red]")
                return
        
        with console.status("[bold green]Generating explanation..."):
            explanation = ai_service.explain_command(command, context)
        
        if explanation:
            console.print(Panel(Markdown(explanation), title="Explanation", expand=False))
        else:
            console.print("[red]❌ Failed to generate explanation[/red]")
            
    except Exception as e:
        logger.error(f"Error explaining command: {e}", exc_info=True)
        console.print(f"[red]❌ Error: {e}[/red]")


def show_history(config: ConfigManager):
    history_manager = HistoryManager()
    history = history_manager.get_recent_history(limit=20)
    
    if not history:
        console.print("[yellow]No command history found[/yellow]")
        return
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim", width=16)
    table.add_column("Query", style="cyan", width=40)
    table.add_column("Command", style="white", width=40)
    table.add_column("✓", justify="center", width=3)
    
    for entry in history:
        executed = "✓" if entry.get("executed") else ""
        time_str = entry.get("timestamp", "")[:16]
        query = entry.get("user_query", "")[:38] + "..." if len(entry.get("user_query", "")) > 40 else entry.get("user_query", "")
        command = entry.get("generated_command", "")[:38] + "..." if len(entry.get("generated_command", "")) > 40 else entry.get("generated_command", "")
        
        table.add_row(time_str, query, command, executed)
    
    console.print(table)
    
    stats = history_manager.get_statistics()
    console.print(f"\n[dim]Total: {stats['total_commands']} | "
                  f"Executed: {stats['executed_commands']} | "
                  f"👍 {stats['positive_feedback']} | "
                  f"👎 {stats['negative_feedback']}[/dim]")


def create_prompt_session(history_manager: HistoryManager) -> PromptSession:
    """Create a prompt-toolkit session with history preloaded from DB."""
    hist = InMemoryHistory()
    try:
        past = history_manager.get_recent_history(limit=200)
        for entry in reversed(past):
            q = entry.get("user_query")
            if q:
                hist.append_string(q)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to preload history: {e}")
    return PromptSession("shwizard> ", history=hist)

def interactive_mode(config: ConfigManager, dry_run: bool = False, safety_enabled: bool = True):
    console.print(Panel.fit(
        "[bold cyan]SHWizard Interactive Mode[/bold cyan]\n"
        "Type your queries in natural language\n"
        "Commands: /help, /history, /stats, /quit",
        border_style="cyan"
    ))
    
    context_manager = ContextManager(
        collect_tools=config.get("context.collect_installed_tools", False)
    )
    history_manager = HistoryManager()
    safety_checker = SafetyChecker(enabled=safety_enabled)
    executor = CommandExecutor(dry_run=dry_run)
    
    ai_service = AIService(
        model=config.get("ollama.model", "gemma3:270m"),
        base_url=config.get("ollama.base_url", "http://localhost:11434")
    )
    
    with console.status("[bold green]Initializing AI service..."):
        if not ai_service.initialize():
            console.print("[red]❌ Failed to initialize AI service[/red]")
            return
    
    console.print("[green]✅ Ready![/green]\n")
    session = create_prompt_session(history_manager)
    
    while True:
        try:
            query = session.prompt().strip()
            
            if not query:
                continue
            
            if query == "/quit" or query == "/exit" or query == "/q":
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if query == "/help":
                show_help()
                continue
            
            if query == "/history":
                show_history(config)
                continue
            
            if query == "/stats":
                stats = history_manager.get_statistics()
                console.print(f"\n📊 [bold]Statistics:[/bold]")
                console.print(f"Total commands: {stats['total_commands']}")
                console.print(f"Executed: {stats['executed_commands']}")
                console.print(f"👍 Positive feedback: {stats['positive_feedback']}")
                console.print(f"👎 Negative feedback: {stats['negative_feedback']}\n")
                continue
            
            process_query(query, config, dry_run, safety_enabled)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            console.print(f"[red]❌ Error: {e}[/red]")


def show_help():
    help_text = """
[bold cyan]SHWizard Help[/bold cyan]

[bold]Commands:[/bold]
  /help      - Show this help message
  /history   - View command history
  /stats     - Show usage statistics
  /quit      - Exit interactive mode

[bold]Usage:[/bold]
  Just type what you want to do in natural language!

[bold]Examples:[/bold]
  - find all python files in current directory
  - show disk usage sorted by size
  - count lines of code in this project
  - compress all images in this folder
"""
    console.print(Panel(help_text, border_style="cyan"))


@click.command()
@click.argument('key')
@click.argument('value', required=False)
def config(key, value):
    config_manager = ConfigManager()
    
    if value:
        config_manager.set(key, value)
        console.print(f"[green]✅ Set {key} = {value}[/green]")
    else:
        current_value = config_manager.get(key)
        console.print(f"[cyan]{key}[/cyan] = [white]{current_value}[/white]")


main.add_command(config)


if __name__ == "__main__":
    main()
