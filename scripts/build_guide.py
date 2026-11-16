"""Assemble docs/LEARNING_GUIDE.md from the chapter files in docs/guide_src/.

Why a build step? The guide quotes a lot of code. Copy-pasted code goes stale
the moment the real file changes, so the chapter sources contain *directives*
instead, and this script replaces them with the current code and line numbers:

    {{code mini_gpt/model.py symbol="CausalSelfAttentionHead.forward"}}
        a whole class / function / method, found by name
    {{code examples/ch02_numpy_basics.py section="4."}}
        from the comment line "# --- 4. ..." up to the next "# --- " line
    {{code train.py from="# 1) Get a batch" to="scaler.update()"}}
        from the first line containing `from` to the next line containing `to`
    {{code configs/small.yaml}}
        the whole file
    {{output examples/ch02_numpy_basics.py from="4. Broadcasting" to="[110. 220. 230.]"}}
        run the script and insert (part of) what it prints
    {{include results/ablations.md from="| Variant" to="| `bpe`"}}
        insert markdown from another file as-is (results tables, samples);
        without `to` it runs to the end of the file; demote="2" turns # into ###

Usage (from the repository root):

    python scripts/build_guide.py            # writes docs/LEARNING_GUIDE.md
    python scripts/build_guide.py --check    # exit with an error if the file is out of date
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "docs" / "guide_src"
OUT_PATH = ROOT / "docs" / "LEARNING_GUIDE.md"

DIRECTIVE = re.compile(r"^\{\{(code|output|include)\s+(\S+?)((?:\s+\w+=\"[^\"]*\")*)\s*\}\}\s*$")
ARGUMENT = re.compile(r"(\w+)=\"([^\"]*)\"")
LANGUAGES = {".py": "python", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown", ".txt": "text"}

_output_cache: dict[str, list[str]] = {}


def find_symbol(source: str, dotted_name: str) -> tuple[int, int]:
    """Return the (first, last) 1-based line numbers of a class/function such as ``GPT.forward``."""
    body = ast.parse(source).body
    node = None
    for part in dotted_name.split("."):
        node = next(
            (n for n in body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part),
            None,
        )
        if node is None:
            raise LookupError(f"symbol {dotted_name!r} not found")
        body = node.body
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])  # include decorators
    return first, node.end_lineno


def find_between(lines: list[str], start_text: str, end_text: str) -> tuple[int, int]:
    """Lines from the first one containing ``start_text`` to the next one containing ``end_text``."""
    first = next((i for i, line in enumerate(lines) if start_text in line), None)
    if first is None:
        raise LookupError(f"start text {start_text!r} not found")
    last = next((i for i in range(first, len(lines)) if end_text in lines[i]), None)
    if last is None:
        raise LookupError(f"end text {end_text!r} not found after {start_text!r}")
    return first + 1, last + 1


def find_section(lines: list[str], label: str) -> tuple[int, int]:
    """Lines from the header ``# --- <label>`` up to (not including) the next ``# --- `` header."""
    first = next((i for i, line in enumerate(lines) if line.startswith(f"# --- {label}")), None)
    if first is None:
        raise LookupError(f"section {label!r} not found")
    last = next((i for i in range(first + 1, len(lines)) if lines[i].startswith("# --- ")), len(lines))
    while last > first and not lines[last - 1].strip():
        last -= 1  # drop trailing blank lines
    return first + 1, last


def select(lines: list[str], source: str, args: dict[str, str]) -> tuple[int, int]:
    """Work out which line range a directive asks for."""
    if "symbol" in args:
        return find_symbol(source, args["symbol"])
    if "section" in args:
        return find_section(lines, args["section"])
    if "from" in args:
        return find_between(lines, args["from"], args.get("to", args["from"]))
    return 1, len(lines)


def render_code(path: str, args: dict[str, str]) -> list[str]:
    """Turn a ``{{code ...}}`` directive into a caption plus a fenced code block."""
    source = (ROOT / path).read_text(encoding="utf-8")
    lines = source.splitlines()
    first, last = select(lines, source, args)
    last = min(len(lines), last + int(args.get("after", "0")))  # after="1": one more line
    where = f"line {first}" if first == last else f"lines {first}–{last}"
    language = LANGUAGES.get(Path(path).suffix, "text")
    return [f"**`{path}` · {where}**", "", f"```{language}", *lines[first - 1 : last], "```"]


def render_output(path: str, args: dict[str, str]) -> list[str]:
    """Turn an ``{{output ...}}`` directive into a fenced block with the script's real output."""
    if path not in _output_cache:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, path], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8"
        )
        if result.returncode != 0:
            raise RuntimeError(f"{path} failed:\n{result.stderr}")
        _output_cache[path] = result.stdout.splitlines()
    lines = _output_cache[path]
    first, last = select(lines, "", args)
    if "from" in args and "to" not in args:
        last = len(lines)  # for output, "from" alone means: from there to the end
    while last > first and not lines[last - 1].strip():
        last -= 1
    return ["```text", *lines[first - 1 : last], "```"]


def render_include(path: str, args: dict[str, str]) -> list[str]:
    """Turn an ``{{include ...}}`` directive into raw markdown taken from another file."""
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    first, last = 1, len(lines)
    if "from" in args:
        first, last = find_between(lines, args["from"], args["to"]) if "to" in args else (
            find_between(lines, args["from"], args["from"])[0], len(lines))
    extra = "#" * int(args.get("demote", "0"))
    out, in_code = [], False
    for line in lines[first - 1 : last]:
        if line.startswith("```"):
            in_code = not in_code
        out.append(extra + line if line.startswith("#") and not in_code else line)
    return out


RENDERERS = {"code": render_code, "output": render_output, "include": render_include}


def expand(text: str, source_name: str) -> str:
    """Replace every directive line in ``text``."""
    out: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = DIRECTIVE.match(line)
        if not match:
            if line.startswith("{{"):
                raise SyntaxError(f"{source_name}:{number}: malformed directive: {line}")
            out.append(line)
            continue
        kind, path, raw_args = match.groups()
        args = dict(ARGUMENT.findall(raw_args))
        try:
            out.extend(RENDERERS[kind](path, args))
        except (LookupError, OSError, RuntimeError) as error:
            raise RuntimeError(f"{source_name}:{number}: {error}") from error
    return "\n".join(out) + "\n"


def table_of_contents(text: str) -> str:
    """A linked list of parts and chapters (GitHub-style heading anchors)."""
    entries = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,2}) (.+)$", line)
        if not match or line.startswith("# The Mini-GPT Learning Guide"):
            continue
        title = match.group(2)
        anchor = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        indent = "" if match.group(1) == "#" else "  "
        entries.append(f"{indent}- [{title}](#{anchor})")
    return "\n".join(entries)


def build() -> str:
    """Return the full text of LEARNING_GUIDE.md."""
    parts = []
    in_code = False
    for path in sorted(SRC_DIR.glob("*.md")):
        parts.append(expand(path.read_text(encoding="utf-8"), path.name))
    text = "\n".join(parts)

    # Headings inside fenced code blocks must not end up in the table of contents.
    visible = []
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        visible.append("" if in_code else line)
    toc = table_of_contents("\n".join(visible))

    header = (
        "<!-- GENERATED FILE: edit the chapters in docs/guide_src/ and run "
        "`python scripts/build_guide.py`. -->\n\n"
    )
    return header + text.replace("<!-- TOC -->", f"<!-- toc:start -->\n{toc}\n<!-- toc:end -->", 1)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Assemble the learning guide.")
    parser.add_argument("--check", action="store_true", help="fail if LEARNING_GUIDE.md is stale")
    args = parser.parse_args()

    text = build()
    if args.check:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != text:
            sys.exit("docs/LEARNING_GUIDE.md is out of date; run: python scripts/build_guide.py")
        print("docs/LEARNING_GUIDE.md is up to date.")
        return
    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    n_snippets = text.count("· line")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} ({len(text.split()):,} words, {n_snippets} code snippets from the repo)")


if __name__ == "__main__":
    main()
