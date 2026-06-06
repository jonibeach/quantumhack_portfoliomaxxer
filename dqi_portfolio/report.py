"""Tiny console-reporting helpers shared by the ``scripts/`` drivers."""

__all__ = ["section"]


def section(title, width=74):
    """Print a titled banner rule (the header every prototype script prints).

    ``width`` keeps the rule length each script historically used (most use 74).
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
