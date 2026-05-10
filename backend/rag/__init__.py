from .search import Hit, search

__all__ = ["Answer", "Hit", "answer", "search"]


def __getattr__(name: str):
    if name in {"Answer", "answer"}:
        from .pipeline import Answer, answer

        return {"Answer": Answer, "answer": answer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
