"""Hard-constraint guards: no runtime AI, no tracked secrets."""
import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BANNED = {"anthropic", "openai", "google.generativeai", "langchain", "cohere", "mistralai"}
BANNED_PACKAGES = {"anthropic", "openai", "google-generativeai", "langchain", "cohere", "mistralai"}


def _project_files():
    return [p for p in [ROOT / "app.py", *sorted((ROOT / "src").rglob("*.py"))] if p.exists()]


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_no_ai_imports_in_project_code():
    bad = [(str(p.relative_to(ROOT)), name) for p in _project_files() for name in _imports(p)
           if any(name == b or name.startswith(b + ".") or name.startswith(b + "_") for b in BANNED)]
    assert not bad, f"AI libraries must not be imported: {bad}"


def test_no_ai_packages_in_requirements():
    names = {re.split(r"[\[<>=!~; ]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")
             for line in (ROOT / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")}
    bad = {n for n in names if any(n == b or n.startswith(b + "-") for b in BANNED_PACKAGES)}
    assert not bad, f"AI packages must not be listed: {bad}"


def test_env_file_is_not_tracked():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git repository yet")
    tracked = subprocess.run(["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True)
    assert tracked.stdout.strip() == "", ".env must never be committed"
    assert ".env" in (ROOT / ".gitignore").read_text().split()
