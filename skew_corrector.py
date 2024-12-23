""" ImageSkewCorrector class """

from typing import Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import cv2
import numpy as np


class ImageSkewCorrector:
    """
    A class for correcting the skew of an image.

    This class provides methods for rotating an image and determining the score of an image.
    """

    def __init__(self):
        # Pre-calculate rotation matrices for common angles
        self._rotation_matrices = {}

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_rotation_matrix(w: int, h: int, angle: float) -> np.ndarray:
        """Cached rotation matrix calculation."""
        center = (w // 2, h // 2)
        return cv2.getRotationMatrix2D(center, angle, 1.0)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Optimized image rotation using cached rotation matrices and minimal memory allocation.
        """
        h, w = image.shape[:2]
        m = self._get_rotation_matrix(w, h, angle)

        # Use faster interpolation for threshold images
        flags = cv2.INTER_NEAREST if image.dtype == np.bool_ else cv2.INTER_LINEAR

        return cv2.warpAffine(
            image, m, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE
        )

    @staticmethod
    def determine_score(arr: np.ndarray) -> np.ndarray:
        """
        Optimized score calculation using vectorized operations.
        """
        # Calculate histogram using optimal dtype
        histogram = np.sum(arr, axis=2, dtype=np.float32)

        # Vectorized difference calculation
        diff = np.diff(histogram, axis=1)
        return np.sum(diff * diff, axis=1, dtype=np.float32)

    def correct_skew(
        self, image: np.ndarray, limit: int = 5, delta: int = 1, num_threads: int = None
    ) -> Tuple[np.ndarray, float]:
        """
        Optimized skew correction with parallel processing and memory efficiency.

        Parameters:
            image: Input image
            limit: Maximum rotation angle
            delta: Angle step size
            num_threads: Number of threads to use (defaults to CPU count)
        """
        if num_threads is None:
            num_threads = min(multiprocessing.cpu_count(), 8)  # Cap at 8 threads

        # Convert to grayscale and threshold more efficiently
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Use faster thresholding for 8-bit images
        if gray.dtype == np.uint8:
            thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]
        else:
            thresh = cv2.threshold(
                (gray * 255).astype(np.uint8),
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1]

        angles = np.arange(-limit, limit + delta, delta)

        # Process images in batches to optimize memory usage
        batch_size = min(len(angles), num_threads * 2)

        def process_batch(batch_angles):
            return np.stack(
                [self.rotate_image(thresh, angle) for angle in batch_angles]
            )

        # Process batches in parallel
        img_list = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(angles), batch_size):
                batch = angles[i : i + batch_size]
                img_list.append(executor.submit(process_batch, batch))

        # Gather results
        img_stack = np.concatenate([future.result() for future in img_list])

        # Compute scores efficiently
        scores = self.determine_score(img_stack)
        best_angle = angles[np.argmax(scores)]

        # Rotate final image with higher quality interpolation
        corrected = self.rotate_image(image, best_angle)
        return corrected, best_angle
