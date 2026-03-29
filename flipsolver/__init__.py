__all__ = ["GridApp"]


def __getattr__(name):
    if name == "GridApp":
        from .gui import GridApp

        return GridApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
