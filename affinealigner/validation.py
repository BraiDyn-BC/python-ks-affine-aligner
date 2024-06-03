# MIT License
#
# Copyright (c) 2024 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Iterable, Optional

import numpy as _np

import affine2d as _affine

from .types import (
    ColorSpec,
    Image,
    ImageMask,
    RGB24Image,
    AffineMatrix,
)
from . import (
    compute as _compute,
)


def overlay_transformed(
    base_image: Image,
    transform: AffineMatrix,
    base_color: ColorSpec = 'm',
    trans_color: ColorSpec = 'c',
) -> RGB24Image:
    trans = _affine.warp_image(base_image, transform)
    base = _compute.color_grayscale(base_image, base_color)
    trans = _compute.color_grayscale(trans, trans_color)
    return _compute.overlay(base, trans)


def generate_borders(
    masks: Iterable[ImageMask],
    transform: Optional[AffineMatrix] = None,
    border_width: int = 2,
) -> ImageMask:
    masks = tuple(masks)
    if len(masks) == 0:
        raise ValueError('no masks to generate borders from')
    borders = _np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        if transform is not None:
            mask = _affine.warp_image(mask, transform).astype(bool)
        border = _compute.mask_to_border(mask, border_width=border_width)
        borders = _np.logical_or(borders, border)
    return borders


def overlay_borders(
    base_image: Image,
    masks: Iterable[ImageMask],
    transform: Optional[AffineMatrix] = None,
    border_width: int = 2,
    base_color: ColorSpec = 'w',
    border_color: ColorSpec = 'w',
    border_alpha: float = 1.0,
) -> RGB24Image:
    if transform is not None:
        base_image = _affine.warp_image(base_image, transform)
    borders = generate_borders(masks, transform=transform, border_width=border_width)
    base = _compute.color_grayscale(base_image, base_color)
    borders = _compute.color_mask(borders, color=border_color, alpha=border_alpha)
    return _compute.overlay(base, borders)
