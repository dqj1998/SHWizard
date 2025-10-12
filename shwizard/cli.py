import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from typing import Optional, List

from shwizard.core.ai_service import AIService
from shwizard.core.context_manager import ContextManager
from shwizard.core.executor import CommandExecutor
from shwizard.safety.checker import SafetyChecker
from shwizard.storage.config import ConfigManager
from shwizard.storage.history import HistoryManager
from shwizard.utils.logger import setup_logger, get_logger

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
    console.print(f"\n🔍 [cyan]Processing:[/cyan] {query}\n")
    
    try:
        context_manager = ContextManager(
            collect_tools=config.get("context.collect_installed_tools", False)
        )
        context = context_manager.get_context()
        
        history_manager = HistoryManager()
        safety_checker = SafetyChecker(enabled=safety_enabled)
        executor = CommandExecutor(dry_run=dry_run)
        
        ai_service = AIService(
            model=config.get("ollama.model", "gemma2:2b"),
            base_url=config.get("ollama.base_url", "http://localhost:11434"),
            timeout=config.get("ollama.timeout", 60)
        )
        
        with console.status("[bold green]Initializing AI service..."):
            if not ai_service.initialize():
                console.print("[red]❌ Failed to initialize AI service[/red]")
                console.print("[yellow]Please ensure Ollama is installed and running[/yellow]")
                return
        
        relevant_commands = []
        if config.get("history.priority_search", True):
            relevant_commands = history_manager.search_relevant_commands(query, limit=5, context=context)
        
        with console.status("[bold green]Generating commands..."):
            commands = ai_service.generate_commands(query, context, relevant_commands)
        
        if not commands:
            console.print("[red]❌ Failed to generate commands[/red]")
            return
        
        console.print(f"[green]✅ Generated {len(commands)} command(s):[/green]\n")
        
        selected_command = select_command(commands, safety_checker, config)
        
        if not selected_command:
            console.print("[yellow]Operation cancelled[/yellow]")
            return
        
        command_id = history_manager.add_command(query, selected_command, context)
        
        if config.get("ui.confirm_execution", True) and not dry_run:
            if not Confirm.ask(f"\n🚀 Execute this command?", default=True):
                console.print("[yellow]Execution cancelled[/yellow]")
                return
        
        success, output = executor.execute(selected_command)
        
        if output:
            console.print("\n[bold]Output:[/bold]")
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


def select_command(commands: List[str], safety_checker: SafetyChecker, config: ConfigManager) -> Optional[str]:
    if len(commands) == 1:
        command = commands[0]
        check_result = safety_checker.check_command(command)
        display_command_with_safety(command, check_result, 1)
        
        if check_result.is_blocked():
            console.print("[red bold]⛔ This command is blocked for safety reasons[/red bold]")
            return None
        
        if check_result.needs_confirmation():
            if not confirm_dangerous_command(command, check_result):
                return None
        
        return command
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("No.", style="dim", width=4)
    table.add_column("Command", style="white")
    table.add_column("Safety", justify="center", width=10)
    
    for idx, cmd in enumerate(commands, 1):
        check_result = safety_checker.check_command(cmd)
        emoji = safety_checker.get_risk_level_emoji(check_result.risk_level)
        color = safety_checker.get_risk_level_color(check_result.risk_level)
        
        table.add_row(
            str(idx),
            cmd,
            f"[{color}]{emoji}[/{color}]"
        )
    
    console.print(table)
    
    while True:
        choice = Prompt.ask(
            "\nSelect command [1-{max}] or 'q' to quit".format(max=len(commands)),
            default="1"
        )
        
        if choice.lower() == 'q':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(commands):
                selected = commands[idx]
                check_result = safety_checker.check_command(selected)
                
                if check_result.is_blocked():
                    console.print("[red bold]⛔ This command is blocked for safety reasons[/red bold]")
                    continue
                
                if check_result.needs_confirmation():
                    if not confirm_dangerous_command(selected, check_result):
                        continue
                
                return selected
            else:
                console.print(f"[red]Please enter a number between 1 and {len(commands)}[/red]")
        except ValueError:
            console.print("[red]Invalid input[/red]")


def display_command_with_safety(command: str, check_result, index: int = 1):
    emoji = "✅" if check_result.is_safe else "⚠️"
    console.print(f"\n{emoji} [bold]Command {index}:[/bold] [white]{command}[/white]")
    
    if not check_result.is_safe or check_result.needs_warning():
        risk_emoji = {"high_risk": "🚨", "medium_risk": "⚠️", "low_risk": "ℹ️"}.get(
            check_result.risk_level, ""
        )
        risk_color = {"high_risk": "red", "medium_risk": "yellow", "low_risk": "blue"}.get(
            check_result.risk_level, "white"
        )
        console.print(f"{risk_emoji} [{risk_color}]Warning: {check_result.message}[/{risk_color}]")


def confirm_dangerous_command(command: str, check_result) -> bool:
    console.print(f"\n[red bold]🚨 DANGEROUS COMMAND DETECTED[/red bold]")
    console.print(f"[yellow]Command:[/yellow] {command}")
    console.print(f"[yellow]Risk:[/yellow] {check_result.message}")
    
    confirmation = Prompt.ask(
        f"\nType 'yes' to proceed or anything else to cancel",
        default="no"
    )
    
    return confirmation.lower() == "yes"


def explain_command(command: str, config: ConfigManager):
    console.print(f"\n🔍 [cyan]Explaining:[/cyan] {command}\n")
    
    try:
        context_manager = ContextManager()
        context = context_manager.get_context()
        
        ai_service = AIService(
            model=config.get("ollama.model", "gemma2:2b"),
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
        model=config.get("ollama.model", "gemma2:2b"),
        base_url=config.get("ollama.base_url", "http://localhost:11434")
    )
    
    with console.status("[bold green]Initializing AI service..."):
        if not ai_service.initialize():
            console.print("[red]❌ Failed to initialize AI service[/red]")
            return
    
    console.print("[green]✅ Ready![/green]\n")
    
    while True:
        try:
            query = Prompt.ask("[bold cyan]shwizard>[/bold cyan]").strip()
            
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
