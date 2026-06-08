#!/usr/bin/env python3
"""
Self-contained Markdown -> standalone HTML builder for the Revizor AArch64 docs.

Single root-of-truth: revizor_aarch64.md  ->  revizor_aarch64.html
Pure Python stdlib only (no pandoc / markdown / pygments needed).

Supports the subset of Markdown this doc uses:
  - ATX headings (#..######) with auto anchors + auto Table of Contents (h2/h3)
  - fenced code blocks ``` (ASCII-art safe; rendered verbatim in <pre>)
  - GitHub pipe tables
  - unordered/ordered lists (one nesting level)
  - blockquotes, horizontal rules
  - inline: `code`, **bold**, *italic*, [text](url)

Usage:  python3 build_docs.py [src.md] [out.html]
"""
import sys, os, re, html

# ----------------------------------------------------------------------------- inline
def _inline(text):
    """Inline formatting on a single line of already-block-stripped text."""
    # protect inline code first
    codes = []
    def stash(m):
        codes.append(html.escape(m.group(1)))
        return f"\x00{len(codes)-1}\x00"
    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", text)
    return text

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

# ----------------------------------------------------------------------------- block
def convert(md):
    lines = md.split("\n")
    out, toc = [], []
    i, n = 0, len(lines)
    slugs = {}

    def uniq(slug):
        if slug not in slugs:
            slugs[slug] = 0; return slug
        slugs[slug] += 1; return f"{slug}-{slugs[slug]}"

    while i < n:
        line = lines[i]

        # fenced code
        if line.startswith("```"):
            i += 1; buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        # heading
        m = re.match(r"(#{1,6})\s+(.*)", line)
        if m:
            lvl = len(m.group(1)); txt = m.group(2).strip()
            slug = uniq(_slug(txt))
            out.append(f'<h{lvl} id="{slug}">{_inline(txt)}</h{lvl}>')
            if lvl in (2, 3):
                toc.append((lvl, txt, slug))
            i += 1; continue

        # horizontal rule
        if re.match(r"^(-{3,}|\*{3,})\s*$", line):
            out.append("<hr>"); i += 1; continue

        # table (header row then |---| separator)
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?\s*:?-{2,}", lines[i+1]) and "|" in lines[i+1]:
            def cells(r): return [c.strip() for c in r.strip().strip("|").split("|")]
            header = cells(line); i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(cells(lines[i])); i += 1
            t = ["<table><thead><tr>"] + [f"<th>{_inline(c)}</th>" for c in header] + ["</tr></thead><tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t)); continue

        # blockquote
        if line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip()); i += 1
            out.append("<blockquote>" + _inline(" ".join(buf)) + "</blockquote>")
            continue

        # lists (one nesting level via 2+ leading spaces)
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            tag = "ol" if re.match(r"^\s*\d+\.\s+", line) else "ul"
            buf = [f"<{tag}>"]; stack = [tag]
            while i < n and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                lm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", lines[i])
                indent = len(lm.group(1))
                if indent >= 2 and len(stack) == 1:
                    sub = "ol" if re.match(r"\d+\.", lm.group(2)) else "ul"
                    buf.append(f"<{sub}>"); stack.append(sub)
                elif indent < 2 and len(stack) == 2:
                    buf.append(f"</{stack.pop()}>")
                buf.append(f"<li>{_inline(lm.group(3))}</li>")
                i += 1
            while stack: buf.append(f"</{stack.pop()}>")
            out.append("".join(buf)); continue

        # blank
        if not line.strip():
            i += 1; continue

        # paragraph (gather consecutive non-structural lines)
        buf = [line]; i += 1
        while i < n and lines[i].strip() and not re.match(r"(#{1,6}\s|```|>|\s*([-*]|\d+\.)\s)", lines[i]) \
                and not re.match(r"^(-{3,}|\*{3,})\s*$", lines[i]):
            buf.append(lines[i]); i += 1
        out.append("<p>" + _inline(" ".join(buf)) + "</p>")

    # build nested TOC
    toc_html = ['<nav class="toc"><div class="toc-title">Contents</div><ul>']
    cur = 2
    for lvl, txt, slug in toc:
        if lvl == 3 and cur == 2: toc_html.append("<ul>")
        if lvl == 2 and cur == 3: toc_html.append("</ul>")
        toc_html.append(f'<li><a href="#{slug}">{html.escape(txt)}</a></li>')
        cur = lvl
    if cur == 3: toc_html.append("</ul>")
    toc_html.append("</ul></nav>")
    return "\n".join(out), "\n".join(toc_html)

# ----------------------------------------------------------------------------- template
CSS = """
:root{--fg:#1b1f24;--muted:#57606a;--accent:#0969da;--bg:#fff;--code:#f6f8fa;--border:#d0d7de;}
*{box-sizing:border-box}
body{margin:0;color:var(--fg);background:var(--bg);font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
.wrap{display:flex;max-width:1200px;margin:0 auto;}
.toc{position:sticky;top:0;align-self:flex-start;width:280px;max-height:100vh;overflow:auto;padding:24px 16px;border-right:1px solid var(--border);font-size:14px;}
.toc-title{font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-size:12px;margin-bottom:8px;}
.toc ul{list-style:none;margin:0;padding-left:12px;}
.toc a{color:var(--fg);text-decoration:none;}
.toc a:hover{color:var(--accent);text-decoration:underline;}
.content{flex:1;min-width:0;padding:32px 40px;}
h1{font-size:2em;border-bottom:2px solid var(--border);padding-bottom:.3em;}
h2{font-size:1.5em;border-bottom:1px solid var(--border);padding-bottom:.3em;margin-top:1.8em;}
h3{font-size:1.2em;margin-top:1.5em;}
code{background:var(--code);padding:.15em .4em;border-radius:6px;font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow:auto;border:1px solid var(--border);}
pre code{background:none;padding:0;font-size:12.5px;line-height:1.45;white-space:pre;}
table{border-collapse:collapse;margin:1em 0;width:100%;font-size:14px;}
th,td{border:1px solid var(--border);padding:6px 10px;text-align:left;vertical-align:top;}
th{background:var(--code);}
blockquote{margin:1em 0;padding:.4em 1em;color:var(--muted);border-left:4px solid var(--border);background:var(--code);}
a{color:var(--accent);}
hr{border:none;border-top:1px solid var(--border);margin:2em 0;}
@media(max-width:900px){.wrap{flex-direction:column}.toc{position:static;width:auto;border-right:none;border-bottom:1px solid var(--border)}}
"""

def page(title, toc, body):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
{toc}
<main class="content">
{body}
</main></div></body></html>
"""

# ----------------------------------------------------------------------------- main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "index.md")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".html"
    text = open(src, encoding="utf-8").read()
    m = re.search(r"^#\s+(.*)", text, re.M)
    title = m.group(1).strip() if m else "Revizor AArch64"

    # Prefer the python-markdown library (robust); fall back to the built-in converter
    # above so the docs always build even with zero extra dependencies installed.
    try:
        import markdown
        md = markdown.Markdown(extensions=["extra", "sane_lists", "toc"],
                               extension_configs={"toc": {"toc_depth": "2-3"}})
        body = md.convert(text)
        toc = f'<nav class="toc"><div class="toc-title">Contents</div>{md.toc}</nav>'
        engine = f"python-markdown {markdown.__version__}"
    except ImportError:
        body, toc = convert(text)
        engine = "built-in converter (markdown lib not installed)"

    open(out, "w", encoding="utf-8").write(page(title, toc, body))
    print(f"built {out} ({os.path.getsize(out)} bytes) from {os.path.basename(src)}  [{engine}]")

if __name__ == "__main__":
    main()
