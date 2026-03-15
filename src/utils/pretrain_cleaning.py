"""Utility helpers for high-throughput pre-training text cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import re
import string
import unicodedata

from typing import Iterable, List

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CONTROL_CHAR_PATTERN = re.compile(r"[\r\x0b\x0c\x0e-\x1f\x7f]")
_DIGIT_PATTERN = re.compile(r"\d")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_HTML_ENTITY_BREAK_PATTERN = re.compile(r"&(?:nbsp|amp|lt|gt|quot);", re.IGNORECASE)
_REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,:;\-_=])\1{2,}")
_REPEATED_CHAR_PATTERN = re.compile(r"([A-Za-z])\1{4,}")
_NON_ALPHA_PATTERN = re.compile(r"[^A-Za-z]+")
_MULTISPACE_PATTERN = re.compile(r"[ \t]+")
_MULTINEWLINE_PATTERN = re.compile(r"\n{3,}")
_WORD_PATTERN = re.compile(
    r"[^\W\d_]+(?:['\u2018\u2019\u02BC\-\u2010\u2011][^\W\d_]+)*",
    re.UNICODE,
)
_COOKIE_PRIVACY_PATTERN = re.compile(
    r"(cookies on .* website|privacy policy|terms of use|accept cookies|all rights reserved)",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the",
    "and",
    "is",
    "in",
    "to",
    "of",
    "that",
    "for",
    "on",
    "with",
    "as",
    "by",
    "this",
    "from",
}


@dataclass(slots=True)
class CleaningConfig:
    """Configuration for source-aware document cleaning."""

    remove_urls: bool = True
    replace_urls_with: str = " "
    remove_emails: bool = True
    replace_emails_with: str = " "
    collapse_whitespace: bool = True
    mask_numbers: bool = False
    number_mask_token: str = "<NUM>"
    remove_control_chars: bool = True
    strip_punct: bool = False
    strip_html: bool = True
    normalize_unicode: bool = True
    normalize_newlines: bool = True
    remove_boilerplate: bool = True
    repair_ocr_lines: bool = True
    collapse_repeated_punctuation: bool = True
    collapse_repeated_characters: bool = True


@dataclass(slots=True)
class QualityConfig:
    """Cheap document-level filters intended for the hot path."""

    min_chars: int = 200
    max_chars: int = 100_000
    min_alpha_ratio: float = 0.55
    max_whitespace_ratio: float = 0.35
    max_symbol_ratio: float = 0.25
    max_repeated_line_ratio: float = 0.4
    max_repeated_ngram_ratio: float = 0.25
    min_stopword_hits: int = 2
    require_english_like: bool = True


@dataclass(slots=True)
class CleanedTextResult:
    """Normalized text plus lightweight quality metadata."""

    normalized_text: str
    dedup_text: str
    quality_flags: List[str] = field(default_factory=list)
    language: str = "unknown"
    char_count: int = 0
    token_estimate: int = 0


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving paragraph boundaries."""
    lines = [_MULTISPACE_PATTERN.sub(" ", line).strip() for line in text.split("\n")]
    collapsed = "\n".join(line for line in lines if line)
    return _MULTINEWLINE_PATTERN.sub("\n\n", collapsed).strip()


def normalize_newlines(text: str) -> str:
    """Convert CRLF and CR newlines to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_urls(text: str, replacement: str = "") -> str:
    """Replace http(s) and www URLs."""
    return _URL_PATTERN.sub(replacement, text)


def remove_emails(text: str, replacement: str = "") -> str:
    """Replace email addresses."""
    return _EMAIL_PATTERN.sub(replacement, text)


def remove_control_characters(text: str) -> str:
    """Drop control characters that interfere with tokenizers."""
    return _CONTROL_CHAR_PATTERN.sub("", text)


def mask_digits(text: str, mask_token: str = "<NUM>") -> str:
    """Replace digits with a mask token while keeping spacing intact."""
    return _DIGIT_PATTERN.sub(mask_token, text)


def strip_punctuation(text: str) -> str:
    """Remove ASCII punctuation characters while preserving whitespace."""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def normalize_unicode_text(text: str) -> str:
    """Normalize compatibility variants for more stable cleaning and dedup."""
    return unicodedata.normalize("NFKC", text)


def strip_html_markup(text: str) -> str:
    """Drop simple HTML tags and decode common entities."""
    text = html.unescape(text)
    text = _HTML_ENTITY_BREAK_PATTERN.sub(" ", text)
    return _HTML_TAG_PATTERN.sub(" ", text)


def remove_pattern_boilerplate(text: str) -> str:
    """Drop lines with obvious web-page boilerplate."""
    kept_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _COOKIE_PRIVACY_PATTERN.search(line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def repair_ocr_line_breaks(text: str) -> str:
    """Join OCR or PDF line wraps when the preceding line is not sentence-final."""
    lines = [line.strip() for line in text.split("\n")]
    repaired: List[str] = []
    for line in lines:
        if not line:
            if repaired and repaired[-1] != "":
                repaired.append("")
            continue
        if repaired and repaired[-1] and repaired[-1][-1] not in ".!?:" and line[:1].islower():
            repaired[-1] = f"{repaired[-1]} {line}"
        else:
            repaired.append(line)
    return "\n".join(repaired)


def collapse_repeated_punctuation_runs(text: str) -> str:
    """Reduce repeated punctuation bursts."""
    return _REPEATED_PUNCT_PATTERN.sub(r"\1\1", text)


def collapse_repeated_character_runs(text: str) -> str:
    """Reduce excessive repeated alphabetic characters."""
    return _REPEATED_CHAR_PATTERN.sub(r"\1\1\1", text)


def repeated_line_ratio(text: str) -> float:
    """Measure how much of the document is repeated line boilerplate."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    repeated = len(lines) - len(set(lines))
    return repeated / len(lines)


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    """Estimate repetition using token n-grams."""
    tokens = _WORD_PATTERN.findall(text.lower())
    if len(tokens) < n:
        return 0.0
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    repeated = len(grams) - len(set(grams))
    return repeated / len(grams)


def estimate_token_count(text: str) -> int:
    """Approximate token count without a tokenizer dependency in the hot path."""
    return len(text.split())


def detect_language_simple(text: str) -> str:
    """Very cheap English-like detector for the first pipeline version."""
    words = _WORD_PATTERN.findall(text.lower())
    if not words:
        return "unknown"
    stopword_hits = sum(1 for word in words if word in _STOPWORDS)
    ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(len(text), 1)
    if stopword_hits >= 2 and ascii_ratio >= 0.9:
        return "en"
    if ascii_ratio < 0.75:
        return "non_en"
    return "unknown"


def make_dedup_text(text: str) -> str:
    """Build a normalized text variant for exact and near dedup."""
    text = text.lower()
    text = strip_punctuation(text)
    text = normalize_whitespace(text)
    return text


def compute_quality_flags(text: str, config: QualityConfig) -> List[str]:
    """Evaluate cheap quality heuristics and return drop reasons."""
    flags: List[str] = []
    if not text:
        return ["empty_text"]

    char_count = len(text)
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    whitespace_chars = sum(1 for ch in text if ch.isspace())
    alnum_chars = sum(1 for ch in text if ch.isalnum())
    symbol_chars = char_count - alnum_chars - whitespace_chars

    alpha_ratio = alpha_chars / char_count
    whitespace_ratio = whitespace_chars / char_count
    symbol_ratio = symbol_chars / char_count
    line_repeat = repeated_line_ratio(text)
    ngram_repeat = repeated_ngram_ratio(text)
    words = _WORD_PATTERN.findall(text.lower())
    stopword_hits = sum(1 for word in words if word in _STOPWORDS)
    language = detect_language_simple(text)

    if char_count < config.min_chars:
        flags.append("too_short")
    if char_count > config.max_chars:
        flags.append("too_long")
    if alpha_ratio < config.min_alpha_ratio:
        flags.append("low_alpha_ratio")
    if whitespace_ratio > config.max_whitespace_ratio:
        flags.append("high_whitespace_ratio")
    if symbol_ratio > config.max_symbol_ratio:
        flags.append("high_symbol_ratio")
    if line_repeat > config.max_repeated_line_ratio:
        flags.append("high_repeated_line_ratio")
    if ngram_repeat > config.max_repeated_ngram_ratio:
        flags.append("high_repeated_ngram_ratio")
    if stopword_hits < config.min_stopword_hits:
        flags.append("low_stopword_hits")
    if config.require_english_like and language not in {"en", "unknown"}:
        flags.append("non_english_like")

    return flags


def clean_pretraining_text(
    text: str,
    *,
    source_type: str = "plain_text",
    cleaning_config: CleaningConfig | None = None,
) -> CleanedTextResult:
    """Normalize a document with source-aware rules for pretraining."""
    config = cleaning_config or CleaningConfig()
    quality_flags: List[str] = []

    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
        quality_flags.append("decoded_with_replacement")
    else:
        text = str(text)

    text = text.strip()
    if config.normalize_unicode:
        text = normalize_unicode_text(text)
    if config.normalize_newlines:
        text = normalize_newlines(text)
    if config.remove_control_chars:
        text = remove_control_characters(text)
    if config.strip_html and source_type in {"html", "web", "plain_text"}:
        text = strip_html_markup(text)
    if config.remove_boilerplate and source_type in {"html", "web", "plain_text"}:
        text = remove_pattern_boilerplate(text)
    if config.remove_urls:
        text = remove_urls(text, replacement=config.replace_urls_with)
    if config.remove_emails:
        text = remove_emails(text, replacement=config.replace_emails_with)
    if source_type in {"pdf_ocr", "ocr"} and config.repair_ocr_lines:
        text = repair_ocr_line_breaks(text)
    if config.collapse_repeated_punctuation:
        text = collapse_repeated_punctuation_runs(text)
    if config.collapse_repeated_characters:
        text = collapse_repeated_character_runs(text)
    if config.strip_punct:
        text = strip_punctuation(text)
    if config.mask_numbers:
        text = mask_digits(text, config.number_mask_token)
    if config.collapse_whitespace:
        text = normalize_whitespace(text)

    language = detect_language_simple(text)
    dedup_text = make_dedup_text(text)
    return CleanedTextResult(
        normalized_text=text,
        dedup_text=dedup_text,
        quality_flags=quality_flags,
        language=language,
        char_count=len(text),
        token_estimate=estimate_token_count(text),
    )


def batch_clean_texts(
    texts: Iterable[str],
    *,
    source_type: str = "plain_text",
    max_output_length: int | None = None,
    cleaning_config: CleaningConfig | None = None,
) -> List[str]:
    """Clean a batch of texts and return normalized text only."""
    cleaned: List[str] = []
    for doc in texts:
        result = clean_pretraining_text(
            doc,
            source_type=source_type,
            cleaning_config=cleaning_config,
        )
        normalized = result.normalized_text
        if max_output_length is not None and len(normalized) > max_output_length:
            normalized = normalized[:max_output_length]
        cleaned.append(normalized)
    return cleaned
