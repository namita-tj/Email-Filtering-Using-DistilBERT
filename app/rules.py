"""Keyword rules: the last resort when the model is unsure and the LLM is unavailable."""

import re
from dataclasses import dataclass

SPAM_PATTERNS: dict[str, re.Pattern] = {
    "prize or lottery": re.compile(r"\b(you('ve| have)? won|winner|lottery|prize|jackpot)\b", re.I),
    "urgent money request": re.compile(r"\b(wire transfer|western union|bitcoin|gift ?cards?)\b", re.I),
    "too-good offer": re.compile(r"\b(100% free|risk[- ]free|no cost|act now|limited time)\b", re.I),
    "credential phishing": re.compile(
        r"\b(verify your (account|password)|account (suspended|locked)|confirm your (bank|login))\b",
        re.I,
    ),
    "quick money": re.compile(r"\b(make money fast|earn \$?\d+[k]? (a|per) (day|week))\b", re.I),
}

SPAM_SCORE_THRESHOLD = 2


@dataclass(frozen=True)
class RuleResult:
    label: str
    matched: tuple[str, ...]

    @property
    def reason(self) -> str:
        if not self.matched:
            return "No spam patterns matched"
        return "Matched spam patterns: " + ", ".join(self.matched)


def classify_with_rules(text: str) -> RuleResult:
    matched = tuple(name for name, pattern in SPAM_PATTERNS.items() if pattern.search(text))
    label = "spam" if len(matched) >= SPAM_SCORE_THRESHOLD else "ham"
    return RuleResult(label=label, matched=matched)
