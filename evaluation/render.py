"""Draws an email with its words highlighted by attribution, as a standalone SVG.

Red means the word pushed towards spam, blue towards ham; stronger colour means
a bigger effect. Uses a monospace font so the layout is predictable everywhere,
including GitHub's README preview.
"""

from collections.abc import Sequence
from xml.sax.saxutils import escape

from app.explain import WordAttribution

WIDTH = 760
PAD = 28
FONT_SIZE = 15
CHAR_W = 9.0  # monospace advance at 15px
LINE_H = 30
SPAM_RGB = (215, 48, 39)
HAM_RGB = (49, 104, 189)


def _fill(weight: float, scale: float) -> str:
    strength = min(abs(weight) / scale, 1.0) if scale else 0.0
    r, g, b = SPAM_RGB if weight > 0 else HAM_RGB
    return f"rgba({r},{g},{b},{0.08 + 0.72 * strength:.3f})" if strength > 0.02 else "none"


def render_svg(
    words: Sequence[WordAttribution],
    label: str,
    spam_probability: float,
    note: str | None = None,
    max_words: int = 80,
) -> str:
    words = list(words)[:max_words]
    scale = max((abs(w.weight) for w in words), default=0.0)

    title = f"Classified as {label} · spam probability {spam_probability:.2f}"
    subtitle = "Integrated Gradients · red pushes towards spam, blue towards ham"

    boxes: list[str] = []
    x, y = PAD, PAD + 78
    max_x = WIDTH - PAD
    for w in words:
        width = len(w.word) * CHAR_W + 10
        if x + width > max_x and x > PAD:
            x, y = PAD, y + LINE_H
        fill = _fill(w.weight, scale)
        boxes.append(
            f'<rect x="{x:.1f}" y="{y - 19}" width="{width:.1f}" height="26" rx="4" fill="{fill}"/>'
            f'<text x="{x + 5:.1f}" y="{y}" class="w">{escape(w.word)}</text>'
        )
        x += width + 6

    height = y + 34 + (24 if note else 0) + PAD // 2
    legend_y = y + 34
    spam_legend = "rgba({},{},{},0.7)".format(*SPAM_RGB)
    ham_legend = "rgba({},{},{},0.7)".format(*HAM_RGB)
    note_svg = f'<text x="{PAD}" y="{legend_y + 22}" class="n">{escape(note)}</text>' if note else ""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" \
viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(title)}">
<style>
.t{{font:600 18px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#1d2327}}
.s{{font:13px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#57606a}}
.w{{font:{FONT_SIZE}px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#1d2327}}
.n{{font:italic 12px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#57606a}}
</style>
<rect width="{WIDTH}" height="{height}" rx="10" fill="#ffffff" stroke="#d0d7de"/>
<text x="{PAD}" y="{PAD + 14}" class="t">{escape(title)}</text>
<text x="{PAD}" y="{PAD + 36}" class="s">{escape(subtitle)}</text>
{"".join(boxes)}
<rect x="{PAD}" y="{legend_y - 10}" width="14" height="14" rx="3" fill="{spam_legend}"/>
<text x="{PAD + 20}" y="{legend_y + 1}" class="s">towards spam</text>
<rect x="{PAD + 130}" y="{legend_y - 10}" width="14" height="14" rx="3" fill="{ham_legend}"/>
<text x="{PAD + 150}" y="{legend_y + 1}" class="s">towards ham</text>
{note_svg}
</svg>
"""
