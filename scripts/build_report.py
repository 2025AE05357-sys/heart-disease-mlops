"""Build the assignment report: docs/report/report.md -> report.html -> report.pdf.

Markdown is converted with python-markdown; formulas are typeset by MathJax and the Mermaid
diagram drawn by mermaid.js (both loaded from jsDelivr), then Chrome prints A4 through Playwright.
Screenshots that do not exist yet are drawn as labelled placeholders and listed at the end, so
the report can be rebuilt as soon as they are saved to docs/screenshots/.

    python scripts/build_report.py      # needs markdown + playwright; drives the installed Chrome
"""

import argparse
import html
import re
import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

CSS = """
@page { size: A4; margin: 16mm 16mm 18mm 16mm; }
body { font-family: "Segoe UI", system-ui, sans-serif; font-size: 10pt; line-height: 1.42;
       color: #1d1d1b; }
h1 { font-size: 18pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; padding-bottom: 2pt; border-bottom: 1px solid #c3c2b7;
     break-after: avoid; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; break-after: avoid; }
p { margin: 4pt 0 6pt; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 3px solid #c8553d;
             background: #f6f3f1; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.6pt;
        break-inside: avoid; }
th, td { border: 1px solid #d6d5ce; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f0efec; }
code { font-family: Consolas, monospace; font-size: 8.8pt; }
pre { background: #f5f5f3; padding: 6pt 8pt; font-size: 8.4pt; break-inside: avoid; }
figure { margin: 8pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 9.5cm; }
/* Screenshots sit two per row, so twenty of them fit the page budget. */
figure.shot { display: inline-block; width: 48.5%; margin: 4pt 0.5% 8pt; vertical-align: top; }
figure.shot img { max-height: 6.2cm; border: 1px solid #d6d5ce; }
figure.shot .pending { min-height: 3.2cm; box-sizing: border-box; }
figcaption { font-size: 8.8pt; color: #52514e; margin-top: 3pt; }
.pending { border: 1.5px dashed #898781; background: #fafaf8; color: #52514e; padding: 16pt 10pt;
           font-size: 9pt; }
.pending strong { color: #1d1d1b; }
pre.mermaid { background: none; text-align: center; padding: 0; margin: 6pt 0 10pt; }
/* A fixed box the SVG scales into (aspect ratio kept); a free-height SVG taller than a page
   combined with break-inside: avoid prints as blank pages. */
pre.mermaid svg { width: 100% !important; max-width: 100% !important; height: 15cm !important; }
mjx-container[display="true"] { margin: 6pt 0 !important; }
"""

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Heart-Disease Risk API: assignment report</title><style>{css}</style>
<script>window.MathJax = {{
  tex: {{ inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] }},
  svg: {{ fontCache: "global" }}, startup: {{ typeset: true }} }};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
</head><body>
"""

TAIL = """
<script>
mermaid.initialize({ startOnLoad: false, theme: "neutral", flowchart: { useMaxWidth: true } });
window.addEventListener("load", () => {
  MathJax.startup.promise
    .then(() => mermaid.run({ querySelector: "pre.mermaid" }))
    .then(() => { window.__ready = true; })
    .catch((e) => { window.__error = String(e); window.__ready = true; });
});
</script></body></html>
"""


def protect(text: str) -> tuple[str, dict[str, str]]:
    """Swap Mermaid blocks and TeX for tokens so Markdown cannot mangle them."""
    stash: dict[str, str] = {}

    def keep(value: str) -> str:
        token = f"@@STASH{len(stash)}@@"
        stash[token] = value
        return token

    text = re.sub(
        r"```mermaid\n(.*?)```",
        lambda m: keep(f'<pre class="mermaid">{html.escape(m.group(1))}</pre>'),
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\$\$(.+?)\$\$", lambda m: keep(f"\\[{html.escape(m.group(1))}\\]"), text, flags=re.S
    )
    text = re.sub(
        r"(?<![\\$])\$([^$\n]+?)\$", lambda m: keep(f"\\({html.escape(m.group(1))}\\)"), text
    )
    return text, stash


def figures(body: str, base: Path) -> tuple[str, list[str]]:
    """Paragraph images become captioned figures; missing files become labelled placeholders."""
    missing: list[str] = []

    def figure(m: re.Match) -> str:
        alt, src = html.unescape(m.group(1)), m.group(2)
        caption = f"<figcaption>{html.escape(alt)}</figcaption>"
        kind = ' class="shot"' if "screenshots/" in src else ""
        if (base / src).exists():
            return f'<figure{kind}><img src="{src}" alt="{html.escape(alt)}">{caption}</figure>'
        missing.append(src)
        name = Path(src).name
        return (
            f'<figure{kind}><div class="pending"><strong>Screenshot pending:</strong> '
            f"docs/screenshots/{html.escape(name)}</div>{caption}</figure>"
        )

    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', figure, body)
    return body, missing


def build(source: Path, keep_html: bool) -> int:
    text = source.read_text(encoding="utf-8")
    text, stash = protect(text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    for token, value in stash.items():
        body = body.replace(f"<p>{token}</p>", value).replace(token, value)
    body, missing = figures(body, source.parent)
    page = HEAD.format(css=CSS) + body + TAIL
    html_path = source.with_suffix(".html")
    pdf_path = source.with_suffix(".pdf")
    html_path.write_text(page, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        tab = browser.new_page()
        tab.goto(html_path.resolve().as_uri())
        tab.wait_for_function("window.__ready === true", timeout=90_000)
        error = tab.evaluate("window.__error || ''")
        tab.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=(
                '<div style="font-size:8px;width:100%;text-align:center;color:#898781;">'
                '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
            ),
        )
        browser.close()
    if not keep_html:
        html_path.unlink()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    print(f"wrote {pdf_path} ({pages} pages)")
    if error:
        print(f"render warning: {error}")
    if missing:
        print(f"{len(missing)} screenshots pending (placeholders in the PDF):")
        for src in missing:
            print(f"  docs/screenshots/{Path(src).name}")
    return 1 if error else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build docs/report/report.pdf")
    parser.add_argument("--source", default="docs/report/report.md")
    parser.add_argument("--keep-html", action="store_true", help="keep report.html next to the PDF")
    args = parser.parse_args(argv)
    return build(Path(args.source), args.keep_html)


if __name__ == "__main__":
    sys.exit(main())
