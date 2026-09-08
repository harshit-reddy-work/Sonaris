"""Concrete preprocessing steps.

Each class has a Single Responsibility. They all implement Preprocessor,
so they are interchangeable (Liskov) and composable in a pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.interfaces import Preprocessor


class MedianDenoiser(Preprocessor):
    """Speckle noise reduction via median filtering."""

    def __init__(self, kernel_size: int = 5) -> None:
        self._kernel_size = kernel_size

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, self._kernel_size)


class NLMDenoiser(Preprocessor):
    """Non-Local Means denoising for fine-grained noise removal."""

    def __init__(self, h: int = 10) -> None:
        self._h = h

    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2 or image.shape[2] == 1:
            return cv2.fastNlMeansDenoising(image, None, self._h)
        return cv2.fastNlMeansDenoisingColored(image, None, self._h)


class CLAHEEnhancer(Preprocessor):
    """Contrast Limited Adaptive Histogram Equalization.

    Sonar shadow-highlight contrast enhancement.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if len(image.shape) == 2:
            return self._clahe.apply(image)

        return self._clahe.apply(image[:, :, 0])


class ImageNormalizer(Preprocessor):
    """Normalize pixel values to [0, 1] float32."""

    def process(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image


class ImageResizer(Preprocessor):
    """Resize image to target dimensions for SAM input."""

    def __init__(self, target_size: tuple[int, int] = (1024, 1024)) -> None:
        self._target_size = target_size

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(
            image,
            self._target_size,
            interpolation=cv2.INTER_LINEAR,
        )


class GrayscaleToRGB(Preprocessor):
    """Convert grayscale image to 3-channel RGB (SAM expects 3 channels)."""

    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
