import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Tuple

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from shwizard.utils.input_utils import (
    COMMON_COMMANDS,
    SIMPLE_COMMANDS,
    SUBCOMMAND_COMMANDS,
    is_command_input,
)
from shwizard.core.ai_service import AIService
from shwizard.storage.history import HistoryManager
from shwizard.storage.config import ConfigManager


def _is_path_like(s: str) -> bool:
    if not s:
        return False
    # Basic path heuristics
    return bool(
        s.startswith(("~", "/", "./", "../"))
        or "/" in s
        or re.search(r"[A-Za-z0-9_\-\/\.]+\.[A-Za-z0-9]{1,6}$", s)
    )


def _safe_run(cmd: List[str], cwd: Optional[str] = None, timeout: float = 0.5) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ,
        )
        return p.returncode, p.stdout.strip()
    except Exception:
        return 1, ""


class ShellCompleter(Completer):
    """
    Context-aware shell completer:
    - Command names: PATH executables + known common commands
    - File/dir paths: list directory entries, append '/' for dirs
    - Subcommands/options: minimal provider for popular CLIs
    - Argument values: light providers (e.g., git branches)
    - LLM-assisted: fallback full command suggestions when static providers yield no results
    """

    def __init__(
        self,
        config: ConfigManager,
        ai_service: Optional[AIService] = None,
        history_manager: Optional[HistoryManager] = None,
    ):
        self.config = config
        self.ai_service = ai_service
        self.history_manager = history_manager
        # Cache executables to avoid rescanning PATH each keystroke
        self._exe_cache: Optional[List[str]] = None
        self._kubectl_resources_cache: Optional[List[str]] = None
        self.case_insensitive = bool(config.get("ui.completion.case_insensitive", True))
        self.llm_enabled = bool(config.get("ui.completion.llm_enabled", True))

    # --------------- Prompt-toolkit hook ---------------
    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        # Parse context
        ctx = self._parse_context(text)

        # Dispatch providers
        items: List[Completion] = []
        if ctx["position"] == "command":
            items = self._complete_command_names(ctx, document)
        elif ctx["position"] == "subcommand":
            items = self._complete_subcommands(ctx, document)
        elif ctx["position"] == "option":
            items = self._complete_options(ctx, document)
        elif ctx["position"] == "path":
            items = self._complete_paths(ctx, document)
        else:  # arg
            items = self._complete_argument_values(ctx, document)

        # LLM-assisted suggestions as fallback (full-line suggestions)
        if self.llm_enabled and not items:
            yield from self._llm_full_line_suggestions(text, document)
            return

        for c in items:
            yield c

    # --------------- Context parsing ---------------
    def _parse_context(self, text: str) -> Dict[str, Optional[str]]:
        """
        Very lightweight parser:
        - tokens_raw: split by whitespace without strict shell quoting
        - current_prefix: last token before cursor (may be empty)
        - command_name: first token if it looks like a command
        - position: one of {"command","subcommand","option","path","arg"}
        """
        s = text or ""
        tokens_raw = s.split()
        # Current prefix = the trailing non-space fragment
        if s.endswith(" ") or not tokens_raw:
            current_prefix = ""
        else:
            current_prefix = tokens_raw[-1]

        # Determine command_name heuristically
        command_name = tokens_raw[0] if tokens_raw else None
        if command_name and not is_command_input(command_name):
            # First token might not be an actual command; treat as None for completion routing
            # But allow common/builtin names
            if command_name not in COMMON_COMMANDS and command_name not in SIMPLE_COMMANDS:
                command_name = None

        # Determine position
        if not tokens_raw or (len(tokens_raw) == 1 and not current_prefix):
            position = "command"
        elif len(tokens_raw) == 1:
            # Still at command position typing a prefix
            position = "command"
        else:
            # More than one token
            if current_prefix.startswith("-"):
                position = "option"
            elif _is_path_like(current_prefix):
                position = "path"
            else:
                # If a known subcommand-style CLI and we're at token index 1 -> subcommand
                if command_name in SUBCOMMAND_COMMANDS and (len(tokens_raw) == 2 and current_prefix):
                    position = "subcommand"
                else:
                    position = "arg"

        return {
            "tokens": tokens_raw,
            "current_prefix": current_prefix,
            "command_name": command_name,
            "position": position,
            "cwd": os.getcwd(),
        }

    # --------------- Providers ---------------
    def _complete_command_names(self, ctx: Dict[str, Optional[str]], document: Document) -> List[Completion]:
        prefix = ctx["current_prefix"] or ""
        candidates = set(COMMON_COMMANDS) | set(SIMPLE_COMMANDS)

        # PATH executables
        if self._exe_cache is None:
            self._exe_cache = self._scan_path_executables()
        candidates.update(self._exe_cache or [])

        return self._match_to_completions(prefix, candidates, document, meta="cmd")

    def _scan_path_executables(self) -> List[str]:
        exes: List[str] = []
        seen = set()
        path = os.environ.get("PATH", "")
        for d in path.split(os.pathsep):
            if not d or not os.path.isdir(d):
                continue
            try:
                for name in os.listdir(d):
                    full = os.path.join(d, name)
                    if name in seen:
                        continue
                    if os.path.isfile(full) and os.access(full, os.X_OK):
                        exes.append(name)
                        seen.add(name)
            except Exception:
                continue
        return sorted(exes)

    def _complete_paths(self, ctx: Dict[str, Optional[str]], document: Document) -> List[Completion]:
        prefix = ctx["current_prefix"] or ""
        base = prefix or ""
        # Expand ~
        expanded = os.path.expanduser(base)
        # Determine directory to list
        if expanded.endswith("/"):
            dir_to_list = expanded
            typed_name = ""
        else:
            dir_to_list = os.path.dirname(expanded) if "/" in expanded else "."
            typed_name = os.path.basename(expanded)

        if not dir_to_list:
            dir_to_list = "."
        try:
            entries = os.listdir(dir_to_list)
        except Exception:
            return []

        completions: List[Completion] = []
        for e in entries:
            if typed_name and not self._match_prefix(e, typed_name):
                continue
            full = os.path.join(dir_to_list, e)
            disp = e
            if os.path.isdir(full):
                # Append slash for directories
                suffix = "/"
                meta = "dir"
            else:
                suffix = ""
                meta = "file"

            # Compute start_position relative to cursor to replace the partial typed_name
            replace_from = -(len(typed_name)) if typed_name else 0
            # Use user-visible path (unexpanded ~ if applicable)
            # For simplicity, insert the expanded path segment
            insertion = e + suffix
            completions.append(
                Completion(
                    insertion,
                    start_position=replace_from,
                    display=disp + suffix,
                    display_meta=meta,
                )
            )
        return sorted(completions, key=lambda c: (c.display_meta or "", c.text.lower()))

    def _complete_subcommands(self, ctx: Dict[str, Optional[str]], document: Document) -> List[Completion]:
        cmd = ctx["command_name"] or ""
        prefix = ctx["current_prefix"] or ""
        sub_map: Dict[str, List[str]] = {
            "git": [
                "add", "commit", "checkout", "switch", "push", "pull", "status",
                "branch", "merge", "rebase", "stash", "log", "tag", "remote",
            ],
            "kubectl": [
                "get", "describe", "apply", "delete", "logs", "exec", "config",
                "cluster-info", "top", "create", "edit", "scale",
            ],
            "npm": ["install", "run", "test", "publish", "ci", "init", "start"],
            "pnpm": ["install", "run", "test", "publish", "dlx"],
            "yarn": ["install", "add", "remove", "run", "test"],
            "docker": ["build", "run", "ps", "images", "pull", "push", "compose"],
            "aws": ["s3", "ec2", "iam", "lambda", "configure"],
            "az": ["login", "account", "group", "vm", "aks", "storage"],
            "brew": ["install", "upgrade", "uninstall", "list", "search"],
            "go": ["build", "run", "test", "mod", "get"],
            "make": [],  # subcommands are targets; leave to LLM/args
            "psql": [],
            "sqlite3": [],
        }
        candidates = sub_map.get(cmd, [])
        return self._match_to_completions(prefix, candidates, document, meta="subcmd")

    def _complete_options(self, ctx: Dict[str, Optional[str]], document: Document) -> List[Completion]:
        cmd = ctx["command_name"] or ""
        prefix = ctx["current_prefix"] or ""
        base_opts = ["-h", "--help", "-v", "--version", "-q", "--quiet", "-y", "--yes"]

        opt_map: Dict[str, List[str]] = {
            "git": ["-a", "--all", "-m", "--message", "--amend", "--no-verify"],
            "kubectl": ["-n", "--namespace", "-o", "--output", "--context"],
            "npm": ["-g", "--global", "--save-dev", "--save"],
            "docker": ["-f", "--file", "-p", "--publish", "--rm"],
        }

        candidates = base_opts + opt_map.get(cmd, [])
        return self._match_to_completions(prefix, candidates, document, meta="opt")

    def _complete_argument_values(self, ctx: Dict[str, Optional[str]], document: Document) -> List[Completion]:
        cmd = ctx["command_name"] or ""
        tokens = ctx["tokens"] or []
        prefix = ctx["current_prefix"] or ""

        # git branch names for checkout/switch
        if cmd == "git" and len(tokens) >= 2 and tokens[1] in ("checkout", "switch"):
            branches = self._git_branches(ctx["cwd"])
            return self._match_to_completions(prefix, branches, document, meta="branch")

        # kubectl common resource names (lightweight default list + optional discovery)
        if cmd == "kubectl":
            res = self._kubectl_resources(ctx["cwd"])
            return self._match_to_completions(prefix, res, document, meta="resource")

        # Fallback: file paths if user started typing a path-like arg
        if _is_path_like(prefix):
            return self._complete_paths(ctx, document)

        # No specific provider; return empty (LLM may fill in)
        return []

    def _git_branches(self, cwd: Optional[str]) -> List[str]:
        code, out = _safe_run(["git", "rev-parse", "--is-inside-work-tree"], cwd=cwd, timeout=0.3)
        if code != 0 or out.strip() != "true":
            return []
        code, out = _safe_run(
            ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads/"],
            cwd=cwd,
            timeout=0.5,
        )
        branches = [b.strip() for b in out.splitlines() if b.strip()]
        return branches

    def _kubectl_resources(self, cwd: Optional[str]) -> List[str]:
        # Use cached discovery if already resolved
        if self._kubectl_resources_cache is not None:
            return self._kubectl_resources_cache
        # Lightweight defaults
        defaults = ["pods", "deployments", "services", "nodes", "namespaces", "configmaps", "secrets"]
        code, out = _safe_run(["kubectl", "api-resources", "-o", "name"], cwd=cwd, timeout=0.6)
        if code != 0 or not out:
            self._kubectl_resources_cache = defaults
            return defaults
        res = [r.strip() for r in out.splitlines() if r.strip()]
        # Deduplicate and cache
        seen = set()
        result = []
        for r in res:
            if r not in seen:
                seen.add(r)
                result.append(r)
        self._kubectl_resources_cache = result
        return result

    # --------------- LLM fallback ---------------
    def _llm_full_line_suggestions(self, text: str, document: Document) -> Iterable[Completion]:
        # Avoid heavy init in the middle of typing; rely on already-initialized AI service
        if not self.ai_service or not self.ai_service.initialize():
            return []
        # Build minimal context (avoid collecting tools to keep latency low)
        context = {
            "os": os.uname().sysname.lower(),
            "cwd": os.getcwd(),
            "shell": os.environ.get("SHELL", "bash").split("/")[-1],
            "installed_tools": [],
        }
        relevant = []
        try:
            if self.history_manager and self.config.get("history.priority_search", True):
                # Provide a few recent commands as hints
                relevant_entries = self.history_manager.search_relevant_commands(text, limit=3, context=context)
                relevant = relevant_entries or []
        except Exception:
            relevant = []

        cmds = self.ai_service.generate_commands(text, context, relevant)
        if not cmds:
            return []

        # Replace entire line with suggested command
        start_pos = -len(document.text_before_cursor)
        for cmd in cmds[:5]:
            yield Completion(cmd, start_position=start_pos, display=cmd, display_meta="llm")

    # --------------- Helpers ---------------
    def _match_prefix(self, candidate: str, prefix: str) -> bool:
        if not prefix:
            return True
        if self.case_insensitive:
            candidate = candidate.lower()
            prefix = prefix.lower()
        return candidate.startswith(prefix)

    def _match_to_completions(self, prefix: str, candidates: Iterable[str], document: Document, meta: str) -> List[Completion]:
        items: List[Completion] = []
        for c in sorted(set(candidates)):
            if self._match_prefix(c, prefix):
                items.append(
                    Completion(
                        text=c,
                        start_position=-(len(prefix)),
                        display=c,
                        display_meta=meta,
                    )
                )
        return items
