# Lightweight i18n utilities
# - Heuristic language detection (fallback)
# - Static translations for CLI UI strings and known safety rule descriptions (fallback)
# - LLM-powered language detection and translation (enhanced, supports many languages)

from typing import Optional, Dict, Tuple
import re

# ------------------------------
# Heuristic detection (fallback)
# ------------------------------

def detect_language(text: Optional[str]) -> str:
    """
    Basic heuristic language detection:
    - Hiragana/Katakana => 'ja'
    - Hangul => 'ko'
    - CJK ideographs (no kana/hangul) => 'zh'
    - else => 'en'
    """
    if not text:
        return "en"
    s = text
    # Japanese: Hiragana or Katakana
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", s):
        return "ja"
    # Korean: Hangul syllables
    if re.search(r"[\uAC00-\uD7AF]", s):
        return "ko"
    # Chinese: CJK ideographs (exclude kana/hangul handled above)
    if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\uF900-\uFAFF]", s):
        return "zh"
    return "en"


def tr(key: str, lang: str, default: Optional[str] = None, **kwargs) -> str:
    """
    Translate a UI string using built-in dictionary only (no LLM).
    """
    s = UI_STRINGS.get(key, {}).get(lang) or UI_STRINGS.get(key, {}).get("en") or default or key
    if kwargs:
        try:
            return s.format(**kwargs)
        except Exception:
            return s
    return s


def translate_rule_description(description: str, lang: str) -> str:
    """
    Translate known rule descriptions using built-in dictionary only (no LLM).
    """
    if lang == "zh":
        return RULE_DESC_ZH.get(description, description)
    # Default: return as-is
    return description


UI_STRINGS = {
    "processing": {
        "en": "Processing",
        "zh": "处理中",
    },
    "blocked_reason": {
        "en": "This command is blocked for safety reasons",
        "zh": "此命令因安全原因被阻止",
    },
    "execution_cancelled": {
        "en": "Execution cancelled",
        "zh": "已取消执行",
    },
    "warning_label": {
        "en": "Warning",
        "zh": "警告",
    },
    "danger_detected_title": {
        "en": "DANGEROUS COMMAND DETECTED",
        "zh": "检测到危险命令",
    },
    "command_label": {
        "en": "Command",
        "zh": "命令",
    },
    "risk_label": {
        "en": "Risk",
        "zh": "风险",
    },
    "type_yes_to_proceed": {
        "en": "Type 'yes' to proceed or anything else to cancel",
        "zh": "输入 'yes' 继续，否则取消",
    },
    "select_command_prompt": {
        "en": "Select command [1-{max}] or 'q' to quit",
        "zh": "选择命令 [1-{max}]，或输入 'q' 退出",
    },
    "selection_hint": {
        "en": "Tip: Enter number to review/edit, or !number (e.g., !1) to execute directly",
        "zh": "提示：输入数字可查看/编辑命令，或输入 !数字 (如 !1) 直接执行",
    },
    "selected_command": {
        "en": "Selected command",
        "zh": "已选择的命令",
    },
    "edit_or_execute_prompt": {
        "en": "Press Enter or 'e' to execute, 'm' to modify, 'q' to cancel",
        "zh": "按 Enter 或 'e' 执行，'m' 修改，'q' 取消",
    },
    "modify_command_prompt": {
        "en": "Modify command",
        "zh": "修改命令",
    },
    "enter_number_between": {
        "en": "Please enter a number between 1 and {max}",
        "zh": "请输入 1 到 {max} 之间的数字",
    },
    "invalid_input": {
        "en": "Invalid input",
        "zh": "无效输入",
    },
    "execute_this_command": {
        "en": "Execute this command?",
        "zh": "执行此命令？",
    },
    "output_label": {
        "en": "Output",
        "zh": "输出",
    },
}

# Known rule descriptions translated to Chinese (keep English source in YAML)
RULE_DESC_ZH = {
    "Delete root directory": "删除根目录",
    "Direct write to disk device": "直接写入磁盘设备",
    "Format filesystem": "格式化文件系统",
    "Fork bomb": "Fork炸弹",
    "Delete root with sudo": "使用sudo删除根目录",
    "Redirect to disk device": "重定向到磁盘设备",
    "Recursive force delete": "递归强制删除",
    "Open all permissions": "开放所有权限",
    "Recursive open all permissions": "递归开放所有权限",
    "Delete with sudo": "使用sudo删除",
    "Download and execute script": "下载并执行脚本",
    "Recursive change owner": "递归更改所有者",
    "Overwrite system config": "覆盖系统配置文件",
    "Move file to /dev/null": "移动文件到/dev/null",
    "Delete files": "删除文件",
    "Use sudo": "使用管理员权限",
}

# ------------------------------
# LLM-powered detection/translation
# ------------------------------

def _normalize_lang(code: Optional[str]) -> str:
    """
    Normalize model-returned language code to ISO 639-1 style.
    Examples:
      'en' -> 'en'
      'en-US' -> 'en'
      '中文' -> 'zh'
    """
    if not code:
        return "en"
    s = code.strip().lower()
    # Map common names to codes
    name_map = {
        "english": "en",
        "chinese": "zh",
        "中文": "zh",
        "日本語": "ja",
        "japanese": "ja",
        "한국어": "ko",
        "korean": "ko",
        "français": "fr",
        "french": "fr",
        "español": "es",
        "spanish": "es",
        "deutsch": "de",
        "german": "de",
        "italiano": "it",
        "italian": "it",
        "português": "pt",
        "portuguese": "pt",
        "русский": "ru",
        "russian": "ru",
        "हिन्दी": "hi",
        "hindi": "hi",
    }
    if s in name_map:
        return name_map[s]
    # Extract first two letters from codes like 'zh-cn', 'en-us'
    m = re.match(r"^([a-z]{2})", s)
    if m:
        return m.group(1)
    # Default to English
    return "en"


class LLMTranslator:
    """
    LLM-backed translator that uses the embedded AIService (Ollama) to:
      - Detect language for arbitrary text
      - Translate arbitrary text to target language
    Includes simple caching and heuristic fallbacks.
    """
    def __init__(self, ai_service):
        self.ai = ai_service
        # Cache: ((text, target_lang)) -> translated_text
        self._translate_cache: Dict[Tuple[str, str], str] = {}
        # Cache: text -> detected_lang
        self._detect_cache: Dict[str, str] = {}

    def detect(self, text: Optional[str]) -> str:
        if not text:
            return "en"
        if text in self._detect_cache:
            return self._detect_cache[text]
        heuristic = detect_language(text)
        detected = heuristic
        # Always try LLM to refine detection when available
        if hasattr(self.ai, "initialize") and hasattr(self.ai, "_call_ollama"):
            try:
                # Ensure model/server are ready
                self.ai.initialize()
                prompt = (
                    "Detect the language of the following text. "
                    "Respond with only the ISO 639-1 language code (e.g., en, zh, ja, ko, fr, es, de, it, pt, ru, hi). "
                    "Text:\n"
                    f"{text}"
                )
                resp = self.ai._call_ollama(prompt, system_prompt="You are a language identification assistant. Output only the code.")
                if resp:
                    llm_code = _normalize_lang(resp)
                    if llm_code:
                        detected = llm_code
            except Exception:
                # Keep heuristic result
                pass
        self._detect_cache[text] = detected
        return detected

    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        if not text:
            return ""
        norm_target = _normalize_lang(target_lang)
        # If same language, return as-is
        if source_lang:
            norm_source = _normalize_lang(source_lang)
            if norm_source == norm_target:
                return text
        key = (text, norm_target)
        if key in self._translate_cache:
            return self._translate_cache[key]
        translated = text
        # Use LLM to translate arbitrary text
        if hasattr(self.ai, "_call_ollama"):
            src = _normalize_lang(source_lang) if source_lang else None
            base_instructions = (
                f"Translate the following text to the target language ({norm_target}). "
                "Return only the translated text, without any explanations, notes, or quotes."
            )
            if src:
                base_instructions += f" The source language is {src}."
            prompt = base_instructions + "\nText:\n" + text
            try:
                resp = self.ai._call_ollama(prompt, system_prompt="You are a professional translator. Return only the translation.")
                if resp:
                    translated = resp.strip()
            except Exception:
                # Fallback: keep original text
                translated = text
        self._translate_cache[key] = translated
        return translated


def tr_llm(key: str, lang: str, translator: Optional[LLMTranslator], default: Optional[str] = None, **kwargs) -> str:
    """
    LLM-aware translation for UI strings:
    - If a localized UI string exists, use it
    - Otherwise, translate from English via LLM to the requested language
    """
    s_en = UI_STRINGS.get(key, {}).get("en") or default or key
    # Apply formatting with kwargs before translation
    if kwargs:
        try:
            s_en = s_en.format(**kwargs)
        except Exception:
            pass
    target = _normalize_lang(lang)
    # If we already have a localized string, return it
    s_local = UI_STRINGS.get(key, {}).get(target)
    if s_local:
        # If formatting placeholders were intended, ensure formatting applied above
        return s_local
    # If target is English, no translation needed
    if target == "en":
        return s_en
    # LLM translation path
    if translator:
        return translator.translate(s_en, target, source_lang="en")
    # Fallback to English
    return s_en


def translate_rule_description_llm(description: str, lang: str, translator: Optional[LLMTranslator]) -> str:
    """
    LLM-aware translation for rule descriptions:
    - If language is Chinese, use static dictionary for consistency
    - Otherwise, use LLM to translate the English description to target language
    """
    target = _normalize_lang(lang)
    if target == "zh":
        return RULE_DESC_ZH.get(description, description)
    if target == "en":
        return description
    if translator:
        return translator.translate(description, target, source_lang="en")
    return description
