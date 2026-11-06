"""Finds words the model may be using as shortcuts instead of real spam signals.

Attributions are aggregated over many validation emails. Two kinds of words are
flagged for a human to review:

- class-exclusive: highly influential words that appear almost only in one
  class of this dataset. They may be dataset artifacts (a sender's signature,
  a template footer) rather than something that generalises.
- non-content: stopwords or one- and two-letter fragments among the most
  influential words. A model leaning on "the" or "s" is not reading content.

Flags are candidates, not verdicts: "lottery" being spam-only is fine.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.explain import WordAttribution

STOPWORDS = frozenset(
    "a an and are as at be by for from has have i in is it of on or our that the this to "
    "was we were will with you your".split()
)


@dataclass
class WordStats:
    word: str
    occurrences: int = 0
    total_weight: float = 0.0
    total_abs_weight: float = 0.0
    spam_emails: int = 0
    ham_emails: int = 0

    @property
    def mean_weight(self) -> float:
        return self.total_weight / self.occurrences if self.occurrences else 0.0

    @property
    def mean_abs_weight(self) -> float:
        return self.total_abs_weight / self.occurrences if self.occurrences else 0.0

    @property
    def emails(self) -> int:
        return self.spam_emails + self.ham_emails

    @property
    def spam_share(self) -> float:
        return self.spam_emails / self.emails if self.emails else 0.0

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "emails": self.emails,
            "mean_weight": round(self.mean_weight, 4),
            "spam_share": round(self.spam_share, 3),
        }


def aggregate(examples: Iterable[tuple[Sequence[WordAttribution], int]]) -> dict[str, WordStats]:
    """`examples` are (word attributions, true label) pairs; label 1 is spam."""
    stats: dict[str, WordStats] = defaultdict(lambda: WordStats(""))
    for words, label in examples:
        seen: set[str] = set()
        for w in words:
            s = stats[w.word]
            s.word = w.word
            s.occurrences += 1
            s.total_weight += w.weight
            s.total_abs_weight += abs(w.weight)
            if w.word not in seen:
                seen.add(w.word)
                if label == 1:
                    s.spam_emails += 1
                else:
                    s.ham_emails += 1
    return dict(stats)


def audit(
    stats: dict[str, WordStats],
    min_emails: int = 5,
    top_n: int = 20,
    exclusive_share: float = 0.95,
) -> dict:
    frequent = [s for s in stats.values() if s.emails >= min_emails]
    by_spam = sorted(frequent, key=lambda s: s.mean_weight, reverse=True)
    by_ham = sorted(frequent, key=lambda s: s.mean_weight)
    influential = sorted(frequent, key=lambda s: s.mean_abs_weight, reverse=True)[:top_n]

    def flag(s: WordStats) -> str | None:
        if s.word in STOPWORDS or len(s.word) <= 2:
            return "non-content"
        if s.spam_share >= exclusive_share or s.spam_share <= 1 - exclusive_share:
            return "class-exclusive"
        return None

    candidates = [{**s.to_dict(), "flag": f} for s in influential if (f := flag(s))]

    return {
        "words_analysed": len(frequent),
        "top_spam_words": [s.to_dict() for s in by_spam[:top_n]],
        "top_ham_words": [s.to_dict() for s in by_ham[:top_n]],
        "shortcut_candidates": candidates,
    }
