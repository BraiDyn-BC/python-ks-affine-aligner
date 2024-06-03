from typing import Tuple, Callable
from collections import namedtuple as _namedtuple

import numpy as _np
import numpy.typing as _npt
import cv2 as _cv2

from .types import (
    Image,
    Number,
    AlignmentMethod,
)
from . import (
    compute as _compute,
)


class FeatureAlignment(
    _namedtuple('FeatureAlignment', ('image1', 'image2', 'kp1', 'desc1', 'kp2', 'desc2', 'matches', 'threshold'))
):
    def as_xy(self) -> Tuple[_npt.NDArray]:
        xy1 = _np.stack([self.kp1[m.queryIdx].pt for m in self.matches if m.distance < self.threshold], axis=0)
        xy2 = _np.stack([self.kp2[m.trainIdx].pt for m in self.matches if m.distance < self.threshold], axis=0)
        return (xy1, xy2)

    def as_image(self) -> Image:
        alignimg = _cv2.drawMatches(
            self.image1,
            self.kp1,
            self.image2,
            self.kp2,
            [m for m in self.matches if m.distance < self.threshold],
            None,
            flags=_cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return alignimg


def align_ORB(
    img1: Image,
    img2: Image,
    scale_factor: Number = 1,
    feature_size: int = 500,
    threshold_factor: float = 0.67,
) -> FeatureAlignment:
    orb = _cv2.ORB_create(feature_size)

    def _convert(img):
        return _compute.to_rgb24(_compute.std_scale(img, scale=scale_factor))

    img1 = _convert(img1)
    img2 = _convert(img2)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = _cv2.BFMatcher(_cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    threshold = min(des1.std(1).min(), des2.std(1).min()) * threshold_factor
    return FeatureAlignment(image1=img1, image2=img2, kp1=kp1, kp2=kp2, desc1=des1, desc2=des2, matches=matches, threshold=threshold)


def get_alignment_method(
    method: AlignmentMethod = 'ORB',
    scale_factor: Number = 1,
    feature_size: int = 500,
    threshold_factor: float = 0.67,
) -> Callable[[Image, Image], FeatureAlignment]:
    if method == 'ORB':
        def _align(img1: Image, img2: Image) -> FeatureAlignment:
            return align_ORB(
                img1,
                img2,
                scale_factor=scale_factor,
                feature_size=feature_size,
                threshold_factor=threshold_factor
            )
    else:
        raise ValueError(f"unexpected method: {repr(method)}")
    return _align
