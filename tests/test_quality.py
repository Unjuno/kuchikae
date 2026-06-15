"""Tests for quality helpers — numeric/negation preservation."""

from __future__ import annotations


from kuchikae.quality import (
    check_negation_preservation,
    check_numeric_preservation,
    extract_negation_markers,
    extract_numeric_tokens,
)


class TestExtractNumericTokens:
    def test_integers(self) -> None:
        assert extract_numeric_tokens("100円です") == ["100円"]

    def test_decimals(self) -> None:
        assert extract_numeric_tokens("3.14") == ["3.14"]

    def test_percentage(self) -> None:
        assert extract_numeric_tokens("50%オフ") == ["50%"]

    def test_date_like(self) -> None:
        tokens = extract_numeric_tokens("2024年12月25日")
        assert "2024年" in tokens
        assert "12月" in tokens
        assert "25日" in tokens

    def test_no_numbers(self) -> None:
        assert extract_numeric_tokens("こんにちは") == []

    def test_negative_number(self) -> None:
        assert extract_numeric_tokens("-5度") == ["-5度"]

    def test_large_unit(self) -> None:
        assert extract_numeric_tokens("1億円") == ["1億円"]

    def test_kanji_numerals(self) -> None:
        assert extract_numeric_tokens("五千万") == ["五千万"]

    def test_kanji_numerals_mixed(self) -> None:
        assert extract_numeric_tokens("一百万円") == ["一百万円"]

    def test_kanji_numerals_with_digits(self) -> None:
        tokens = extract_numeric_tokens("10億円と五千万円")
        assert "10億円" in tokens
        assert "五千万円" in tokens


class TestExtractNegationMarkers:
    def test_japanese_negation(self) -> None:
        markers = extract_negation_markers("食べない")
        assert "ない" in markers

    def test_japanese_masen(self) -> None:
        assert "ません" in extract_negation_markers("行きません")

    def test_english_negation(self) -> None:
        assert "not" in extract_negation_markers("not available")

    def test_no_negation(self) -> None:
        assert extract_negation_markers("hello world") == []

    def test_multiple_markers(self) -> None:
        markers = extract_negation_markers("not available: 不要です")
        assert "not" in markers
        assert "不要" in markers


class TestCheckNumericPreservation:
    def test_preserved(self) -> None:
        assert check_numeric_preservation("100円", "合計100円")

    def test_not_preserved(self) -> None:
        assert not check_numeric_preservation("100円", "合計です")

    def test_no_numbers(self) -> None:
        assert check_numeric_preservation("hello", "world")

    def test_exact_match(self) -> None:
        assert check_numeric_preservation("100円, 200円", "200円, 100円", require_exact=True)

    def test_exact_mismatch(self) -> None:
        assert not check_numeric_preservation("100円", "100円, 200円", require_exact=True)


class TestCheckNegationPreservation:
    def test_preserved(self) -> None:
        assert check_negation_preservation("行きません", "行きませんでした")

    def test_not_preserved(self) -> None:
        assert not check_negation_preservation("行きません", "行きます")

    def test_no_negation(self) -> None:
        assert check_negation_preservation("hello", "world")

    def test_case_insensitive(self) -> None:
        assert check_negation_preservation("Not", "not available")
