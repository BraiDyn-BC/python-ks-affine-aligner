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

from typing import Optional, Iterable

import numpy as _np
import numpy.typing as _npt
import scipy.ndimage as _ndimage
import matplotlib.colors as _mcolors

from .types import (
    ColorSpec,
    Image,
    FloatGrayscaleImage,
    Uint8Image,
    RGB24Image,
    ImageMask,
    Number,
    Point,
    Coordinates,
)


def get_rgbarray(color: ColorSpec) -> _npt.NDArray[_np.float32]:
    if isinstance(color, str):
        color = _mcolors.to_rgb(color)
    return _np.array(color, dtype=_np.float32).reshape((1, 1, 3))


def std_scale(
    img: Image,
    scale: Number = 1,
) -> Image:
    return (img.astype(_np.float32) + scale) / (scale * 2)


def scale_image(
    img: Image,
    vmin: Optional[Number] = None,
    vmax: Optional[Number] = None,
) -> Image:
    img = img.astype(_np.float32)
    if vmin is None:
        vmin = _np.nanmin(img)
    if vmax is None:
        vmax = _np.nanmax(img)
    return (img - vmin) / (vmax - vmin)


def overlay(*images: Iterable[RGB24Image]) -> RGB24Image:
    images = tuple(images)
    if len(images) == 0:
        raise ValueError('no images to overlay')
    elif len(images) == 1:
        return images[0]
    else:
        overlaid = _np.zeros(images[0].shape, dtype=_np.float32)
    for img in images:
        overlaid[:] = overlaid + img.astype(_np.float32)
    overlaid[overlaid < 0] = 0
    overlaid[overlaid > 255] = 255
    return to_rgb24(overlaid, factor=1)


def to_float_grayscale(img: Image) -> FloatGrayscaleImage:
    img = img.astype(_np.float32)
    K = _np.ndim(img)
    if K > 2:
        img = img.mean(_np.arange(2, K))
    return img


def to_uint8(img: Image, factor: Number = 255) -> Uint8Image:
    if img.dtype == _np.uint8:
        return img
    img = img.astype(_np.float32) * factor
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(_np.uint8)


def to_rgb24(img: Image, factor: Number = 255) -> RGB24Image:
    if _np.ndim(img) == 3:
        if img.dtype == _np.uint8:
            return img
        else:
            return to_uint8(img, factor=factor)
    else:
        if img.dtype != _np.uint8:
            img = to_uint8(img, factor=factor)
        return _np.stack([img, img, img], axis=2)


def color_grayscale(
    img: FloatGrayscaleImage,
    color: ColorSpec,
    vmin: Optional[Number] = None,
    vmax: Optional[Number] = None,
) -> RGB24Image:
    img = scale_image(img, vmin=vmin, vmax=vmax)
    if _np.ndim(img) != 3:
        img = img.reshape(img.shape + (1,))
    return to_rgb24(img * get_rgbarray(color), factor=255)


def color_mask(
    mask: ImageMask,
    color: ColorSpec,
    alpha: float = 1.0,
) -> RGB24Image:
    mask = mask.astype(_np.float32).reshape(mask.shape + (1,)) * get_rgbarray(color) * alpha
    return to_rgb24(mask, factor=255)


def mask_to_border(
    mask: ImageMask,
    border_width: int = 2,
) -> ImageMask:
    border_width = int(border_width)
    unit = border_width // 2
    mask = (mask > 0)
    outer = _ndimage.binary_dilation(mask, iterations=unit)
    inner = _ndimage.binary_erosion(mask, iterations=border_width - unit)
    return _np.logical_xor(outer, inner)


def center_of_mass(
    img: Image,
    coords: Coordinates = None,
    as_int: bool = True
) -> Point:
    img = to_float_grayscale(img)
    if coords is None:
        coords = Coordinates.from_image(img)
    weights = img.sum()
    ci = (coords.I * img).sum() / weights
    cj = (coords.J * img).sum() / weights
    if as_int == True:
        ci = round(ci)
        cj = round(cj)
    return Point(ci, cj)


def subtract_background(img: Image, smoothing_dia: Number = 7) -> Image:
    img = to_float_grayscale(img)
    zimg = (img - img.mean(keepdims=True)) / img.std(keepdims=True)
    back = _ndimage.gaussian_filter(zimg, smoothing_dia)
    return zimg - back
