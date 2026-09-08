"""Preprocessing pipeline that composes multiple Preprocessor steps.

Open/Closed: Add new steps without modifying this class.
Dependency Inversion: Depends on Preprocessor interface, not concrete classes.
"""

from __future__ import annotations

import numpy as np

from ..core.interfaces import Preprocessor


class PreprocessingPipeline:
    """Executes a chain of Preprocessor steps in order."""

    def __init__(self, steps: list[Preprocessor] | None = None) -> None:
        self._steps: list[Preprocessor] = steps or []

    def add_step(self, step: Preprocessor) -> PreprocessingPipeline:
        """Fluent API for adding steps."""
        self._steps.append(step)
        return self

    def run(self, image: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps sequentially."""
        result = image.copy()
        for step in self._steps:
            result = step.process(result)
        return result
