from __future__ import annotations

import unittest

from src.utils.pretrain_cleaning import (
    _WORD_PATTERN,
    CleaningConfig,
    QualityConfig,
    clean_pretraining_text,
    compute_quality_flags,
    repair_ocr_line_breaks,
    repeated_ngram_ratio,
)


class PretrainCleaningTests(unittest.TestCase):
    def test_unicode_and_url_cleanup(self) -> None:
        result = clean_pretraining_text(
            "Cafe\u0301 visit <b>today</b> at https://example.com",
            source_type="web",
            cleaning_config=CleaningConfig(),
        )
        self.assertIn("Café visit today", result.normalized_text)
        self.assertNotIn("https://example.com", result.normalized_text)

    def test_ocr_line_repair(self) -> None:
        text = "This is a broken line\ncontinuation text.\n\nNew paragraph."
        repaired = repair_ocr_line_breaks(text)
        self.assertIn("broken line continuation text.", repaired)

    def test_quality_flags_for_short_noisy_text(self) -> None:
        flags = compute_quality_flags("!!! $$$", QualityConfig(min_chars=5))
        self.assertIn("low_alpha_ratio", flags)
        self.assertIn("high_symbol_ratio", flags)

    def test_repeated_ngram_ratio_handles_unicode_and_hyphenated_words(self) -> None:
        text = "Café-au-lait is nice. Café-au-lait is nice."
        ratio = repeated_ngram_ratio(text, n=3)
        self.assertGreater(ratio, 0.0)

    def test_word_pattern_keeps_curly_apostrophes_and_unicode_hyphens(self) -> None:
        tokens = _WORD_PATTERN.findall("don’t l’artiste co‐operate")
        self.assertIn("don’t", tokens)
        self.assertIn("l’artiste", tokens)
        self.assertIn("co‐operate", tokens)


if __name__ == "__main__":
    unittest.main()
