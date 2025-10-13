import sys
import os
import shlex
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from typing import Optional, List, Tuple
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
try:
    from prompt_toolkit.completion import CompleteStyle  # type: ignore
except Exception:
    CompleteStyle = None  # type: ignore
from prompt_toolkit.key_binding import KeyBindings
try:
    from prompt_toolkit.enums import EditingMode  # type: ignore
except Exception:
    EditingMode = None  # type: ignore

from shwizard.core.ai_service import AIService
from shwizard.core.context_manager import ContextManager
from shwizard.core.executor import CommandExecutor
from shwizard.safety.checker import SafetyChecker
from shwizard.storage.config import ConfigManager
from shwizard.storage.history import HistoryManager
from shwizard.utils.logger import setup_logger, get_logger
from shwizard.utils.input_utils import is_command_input
from shwizard.utils.i18n import LLMTranslator, tr_llm, translate_rule_description_llm
from shwizard.utils.completion import ShellCompleter
from shwizard.utils.output_utils import sanitize_output

console = Console()
logger = None

PREV_DIR = None

def set_mouse_mode(enabled: bool):
    """
    Explicitly toggle terminal mouse reporting via ANSI escape sequences.
    Some terminals (including VSCode integrated terminal) ignore application-level toggles;
    sending these sequences ensures native selection can work when disabled.
    """
    try:
        if enabled:
            # Enable minimal mouse reporting for prompt usage (avoid 1002/1003 which can be more invasive)
            sys.stdout.write("\x1b[?1000h\x1b[?1006h")
        else:
            # Disable all known mouse tracking modes to restore native selection in terminals
            sys.stdout.write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1001l\x1b[?1015l")
        sys.stdout.flush()
    except Exception as e:
        if logger:
            logger.warning(f"Failed to set terminal mouse mode: {e}")


def init_logger(config: ConfigManager):
    global logger
    logger = setup_logger(
        level=config.get("logging.level", "INFO"),
        log_file=config.get("logging.file"),
        max_size_mb=config.get("logging.max_size_mb", 10),
        backup_count=config.get("logging.backup_count", 3)
    )
    return logger

def handle_cd(command_str: str, tokens: List[str], history_manager, context, lang: str, translator: LLMTranslator):
    """
    Handle 'cd' inside SHWizard interactive session by changing Python process cwd,
    so subsequent commands run in the new directory. Supports: cd, cd <path>, cd ~, cd -, cd ../rel.
    """
    global logger, PREV_DIR
    cwd_before = os.getcwd()

    # Determine target directory
    arg = tokens[1] if len(tokens) > 1 else None
    if arg is None or arg == "~":
        target = os.path.expanduser("~")
    elif arg == "-":
        target = PREV_DIR or os.environ.get("OLDPWD") or cwd_before
    else:
        expanded = os.path.expanduser(arg)
        target = expanded if os.path.isabs(expanded) else os.path.abspath(os.path.join(cwd_before, expanded))

    success = False
    message = ""
    if target and os.path.isdir(target):
        try:
            os.chdir(target)
            os.environ["OLDPWD"] = cwd_before
            PREV_DIR = cwd_before
            success = True
            message = f"Changed directory to: {os.getcwd()}"
            console.print(f"[green]✅ {message}[/green]")
        except Exception as e:
            message = f"Failed to change directory: {e}"
            console.print(f"[red]❌ {message}[/red]")
    else:
        message = f"No such directory: {target}" if target else "Invalid path"
        console.print(f"[red]❌ {message}[/red]")

    try:
        command_id = history_manager.add_command(command_str, command_str, context)
        history_manager.mark_executed(command_id, success, message)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to record cd command history: {e}")


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
        interactive_mode(config, dry_run, not no_safety)
        return
    
    query_text = " ".join(query)
    
    if explain:
        explain_command(query_text, config)
    else:
        process_query(query_text, config, dry_run, not no_safety)


def process_query(query: str, config: ConfigManager, dry_run: bool = False, safety_enabled: bool = True):
    ai_service = AIService(
        model=config.get("ollama.model", "gemma3:270m"),
        base_url=config.get("ollama.base_url", "http://localhost:11435"),
        timeout=config.get("ollama.timeout", 60)
    )
    translator = LLMTranslator(ai_service)

    # Language preference and detection logic:
    # - Load preferred language (for command-like inputs)
    # - Detect language for natural language queries and persist it
    history_manager = HistoryManager()
    preferred_lang = history_manager.get_preferred_language("en")
    is_cmd = is_command_input(query)
    # Only detect language for natural-language queries to avoid misclassification on raw shell commands
    if is_cmd:
        lang = preferred_lang or "en"
    else:
        detected_lang = translator.detect(query)
        lang = detected_lang
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
            # Handle built-in commands that affect session state (e.g., cd)
            try:
                tokens = shlex.split(query)
            except Exception:
                tokens = query.split()
            if tokens and tokens[0] == "cd":
                handle_cd(query, tokens, history_manager, context, lang, translator)
                return
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
                _san = sanitize_output(output)
                _text = Text.from_ansi(_san)
                console.print(Panel(_text, expand=True))
            
            history_manager.mark_executed(command_id, success, output[:500] if output else None)
            
            # Skip post-execution feedback prompts for direct commands
            return
        
        # Natural language path: use AI to generate commands
        with ai_service:
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
                _san = sanitize_output(output)
                _text = Text.from_ansi(_san)
                console.print(Panel(_text, expand=True))
            
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
        
        with AIService(
            model=config.get("ollama.model", "gemma3:270m"),
            base_url=config.get("ollama.base_url", "http://localhost:11435")
        ) as ai_service:
            
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


def create_prompt_session(config: ConfigManager, history_manager: HistoryManager, ai_service: AIService) -> PromptSession:
    """Create a prompt-toolkit session with history preloaded from DB and a context-aware completer."""
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
    completer = ShellCompleter(config=config, ai_service=ai_service, history_manager=history_manager) if config.get("ui.completion.enabled", True) else None

    # Key bindings to support Home/End keys
    kb = KeyBindings()

    @kb.add("home")
    def _(event):
        buf = event.current_buffer
        buf.cursor_position += buf.document.get_start_of_line_position()

    @kb.add("end")
    def _(event):
        buf = event.current_buffer
        buf.cursor_position += buf.document.get_end_of_line_position()

    # Quick toggle mouse support with F9: exit prompt with sentinel so loop restarts with new setting
    @kb.add("f9")
    def _(event):
        try:
            current = bool(config.get("ui.mouse_support", True))
            config.set("ui.mouse_support", not current)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to toggle mouse_support via F9: {e}")
        # Exit current prompt and return a sentinel
        event.app.exit(result="__MOUSE_TOGGLE__")

    # Optional editing mode from config: "emacs" or "vi"
    editing_mode_key = str(config.get("ui.editing_mode", "emacs")).lower()
    editing_mode = None
    if EditingMode is not None:
        if editing_mode_key == "vi" and hasattr(EditingMode, "VI"):
            editing_mode = EditingMode.VI
        elif hasattr(EditingMode, "EMACS"):
            editing_mode = EditingMode.EMACS

    if editing_mode is not None:
        return PromptSession("shwizard> ", history=hist, completer=completer, key_bindings=kb, editing_mode=editing_mode)
    else:
        return PromptSession("shwizard> ", history=hist, completer=completer, key_bindings=kb)

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
    
    with AIService(
        model=config.get("ollama.model", "gemma3:270m"),
        base_url=config.get("ollama.base_url", "http://localhost:11435")
    ) as ai_service:
        
        with console.status("[bold green]Initializing AI service..."):
            if not ai_service.initialize():
                console.print("[red]❌ Failed to initialize AI service[/red]")
                return
        
        console.print("[green]✅ Ready![/green]\n")
        session = create_prompt_session(config, history_manager, ai_service)
        # Ensure native selection works by default
        set_mouse_mode(False)

        # Completion behavior from config
        style_key = config.get("ui.completion.style", "readline_like")
        complete_style = None
        if 'CompleteStyle' in globals() and CompleteStyle is not None:
            if style_key == "multi_column" and hasattr(CompleteStyle, "MULTI_COLUMN"):
                complete_style = CompleteStyle.MULTI_COLUMN
            elif style_key == "column" and hasattr(CompleteStyle, "COLUMN"):
                complete_style = CompleteStyle.COLUMN
            elif hasattr(CompleteStyle, "READLINE_LIKE"):
                complete_style = CompleteStyle.READLINE_LIKE
        complete_while = bool(config.get("ui.completion.complete_while_typing", False))
        complete_in_thread = bool(config.get("ui.completion.complete_in_thread", True))
        
        while True:
            try:
                auto_mouse = bool(config.get("ui.mouse_auto_mode", True))
                mouse_support = bool(config.get("ui.mouse_support", True))
                # Explicitly set terminal mouse reporting to match config
                set_mouse_mode(mouse_support)
                try:
                    if complete_style is not None:
                        query = session.prompt(
                            complete_while_typing=complete_while,
                            complete_in_thread=complete_in_thread,
                            mouse_support=mouse_support,
                            complete_style=complete_style
                        ).strip()
                    else:
                        query = session.prompt(
                            complete_while_typing=complete_while,
                            complete_in_thread=complete_in_thread,
                            mouse_support=mouse_support
                        ).strip()
                except TypeError:
                    # Fallback for older prompt_toolkit versions that don't support complete_style
                    query = session.prompt(
                        complete_while_typing=complete_while,
                        complete_in_thread=complete_in_thread,
                        mouse_support=mouse_support
                    ).strip()
                
                if query == "__MOUSE_TOGGLE__":
                    new_state = bool(config.get("ui.mouse_support", True))
                    state = "enabled" if new_state else "disabled"
                    console.print(f"[green]Mouse support {state}[/green]")
                    console.print("[dim]Tip: With mouse support enabled, some terminals require holding Shift to select text. Use /mouse off to select freely.[/dim]")
                    set_mouse_mode(new_state)
                    continue
                if not query:
                    continue
                
                if query == "/quit" or query == "/exit" or query == "/q":
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if query == "/help":
                    show_help()
                    if bool(config.get("ui.mouse_auto_mode", True)):
                        try:
                            config.set("ui.mouse_support", False)
                            set_mouse_mode(False)
                        except Exception as e:
                            if logger:
                                logger.warning(f"Failed to disable mouse_support in auto mode: {e}")
                    continue
                
                if query == "/history":
                    show_history(config)
                    if bool(config.get("ui.mouse_auto_mode", True)):
                        try:
                            config.set("ui.mouse_support", False)
                            set_mouse_mode(False)
                        except Exception as e:
                            if logger:
                                logger.warning(f"Failed to disable mouse_support in auto mode: {e}")
                    continue
                
                if query == "/stats":
                    stats = history_manager.get_statistics()
                    console.print(f"\n📊 [bold]Statistics:[/bold]")
                    console.print(f"Total commands: {stats['total_commands']}")
                    console.print(f"Executed: {stats['executed_commands']}")
                    console.print(f"👍 Positive feedback: {stats['positive_feedback']}")
                    console.print(f"👎 Negative feedback: {stats['negative_feedback']}\n")
                    if bool(config.get("ui.mouse_auto_mode", True)):
                        try:
                            config.set("ui.mouse_support", False)
                            set_mouse_mode(False)
                        except Exception as e:
                            if logger:
                                logger.warning(f"Failed to disable mouse_support in auto mode: {e}")
                    continue

                # Toggle mouse support to allow selecting/copying previous outputs when off
                if query.startswith("/mouse"):
                    parts = query.split()
                    if len(parts) == 1 or parts[1].lower() == "toggle":
                        mouse_support = not mouse_support
                    elif parts[1].lower() in ("on", "off"):
                        mouse_support = parts[1].lower() == "on"
                    else:
                        console.print("[yellow]Usage: /mouse [on|off|toggle][/yellow]")
                        continue
                    # Persist preference
                    try:
                        config.set("ui.mouse_support", mouse_support)
                    except Exception as e:
                        if logger:
                            logger.warning(f"Failed to persist mouse_support: {e}")
                    state = "enabled" if mouse_support else "disabled"
                    console.print(f"[green]Mouse support {state}[/green]")
                    console.print("[dim]Tip: With mouse support enabled, some terminals require holding Shift to select text. Use /mouse off to select freely.[/dim]")
                    set_mouse_mode(mouse_support)
                    continue

                # Auto mode: enable/disable mouse by state
                if query.startswith("/mouse-auto"):
                    parts = query.split()
                    if len(parts) == 1:
                        auto_state = bool(config.get("ui.mouse_auto_mode", True))
                        console.print(f"[cyan]Auto mouse mode is {'on' if auto_state else 'off'}[/cyan]")
                        continue
                    val = parts[1].lower()
                    if val in ("on", "off"):
                        auto_state = val == "on"
                        try:
                            config.set("ui.mouse_auto_mode", auto_state)
                            if auto_state:
                                # Default to mouse off immediately so selection works now; enable manually with F9 or /mouse on
                                config.set("ui.mouse_support", False)
                                set_mouse_mode(False)
                        except Exception as e:
                            if logger:
                                logger.warning(f"Failed to persist mouse_auto_mode: {e}")
                        console.print(f"[green]Auto mouse mode {'enabled' if auto_state else 'disabled'}[/green]")
                        if auto_state:
                            console.print("[dim]Auto mode: Mouse is OFF after output for easy selection; use F9 or /mouse on when you need click-to-move.[/dim]")
                        continue
                    else:
                        console.print("[yellow]Usage: /mouse-auto [on|off][/yellow]")
                        continue
                
                process_query(query, config, dry_run, safety_enabled)
                console.print()
                # Auto mode: after output, disable mouse to allow selection
                if bool(config.get("ui.mouse_auto_mode", True)):
                    try:
                        config.set("ui.mouse_support", False)
                        set_mouse_mode(False)
                    except Exception as e:
                        if logger:
                            logger.warning(f"Failed to disable mouse_support in auto mode: {e}")
                
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
  /mouse     - Toggle mouse support: /mouse [on|off|toggle]
  /mouse-auto - Auto mouse mode: off after output for easy selection; toggle on with F9 or /mouse. Usage: /mouse-auto [on|off]
  /quit      - Exit interactive mode

[bold]Usage:[/bold]
  Just type what you want to do in natural language!
  Press F9 to quickly toggle mouse support

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
