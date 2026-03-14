"""
scripts/verify_imports.py

Smoke test — verifies every runtime dependency imports correctly.
Run after any environment change:

    uv run python scripts/verify_imports.py
"""

import sys

CHECKS: list[tuple[str, str]] = [
    # (import_name, display_name)
    ("pydantic",               "pydantic"),
    ("pydantic_settings",      "pydantic-settings"),
    ("httpx",                  "httpx"),
    ("apscheduler",            "apscheduler"),
    ("watchdog",               "watchdog"),
    ("typer",                  "typer"),
    ("prompt_toolkit",         "prompt-toolkit"),
    ("rich",                   "rich"),
    ("structlog",              "structlog"),
    ("faiss",                  "faiss-cpu"),
    ("sentence_transformers",  "sentence-transformers"),
    ("networkx",               "networkx"),
]

DEV_CHECKS: list[tuple[str, str]] = [
    ("pytest",   "pytest"),
    ("mypy",     "mypy"),
    ("ruff",     "ruff"),
]


def check(import_name: str, display: str) -> bool:
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        print(f"  [ok]  {display:<28} {version}")
        return True
    except ImportError as e:
        print(f"  [!!]  {display:<28} MISSING — {e}")
        return False


def main() -> None:
    print("\nRuntime dependencies")
    print("─" * 50)
    results = [check(name, display) for name, display in CHECKS]

    print("\nDev dependencies")
    print("─" * 50)
    dev_results = [check(name, display) for name, display in DEV_CHECKS]

    total = len(results) + len(dev_results)
    passed = sum(results) + sum(dev_results)
    failed = total - passed

    print("\n" + "─" * 50)
    if failed == 0:
        print(f"  All {total} imports OK — environment is ready.\n")
    else:
        print(f"  {passed}/{total} passed — {failed} missing.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()