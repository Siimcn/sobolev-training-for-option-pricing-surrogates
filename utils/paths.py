"""
Where the project writes its output.
"""

from pathlib import Path

__all__ = ["PROJECT_ROOT", "project_path"]

# utils/paths.py -> utils -> repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts) -> Path:
    """
    A path under the repository root.
    """

    return PROJECT_ROOT.joinpath(*parts)
