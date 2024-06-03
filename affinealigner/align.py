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
import math as _math

import numpy as _np

import affine2d as _affine

from . import (
    featurebased as _featurebased,
    compute as _compute,
)
from .types import (
    Number,
    Image,
    Point,
    Coordinates,
    AlignmentMethod,
    AffineMatrix,
)


def align_images(
    images: Iterable[Image],
    use_percentile: int = 15,
    background_dia: Number = 7,
    scale_factor: Number = 1.8,
    alignment_method: AlignmentMethod = 'ORB',
    feature_size: Optional[int] = None,
    threshold_factor: float = 0.67,
) -> Iterable[AffineMatrix]:
    """perform alignment of images.

    parameters
    ----------
    images: an iterable of images in shape (H, W)
            it is assumed that the top-bottom axis
            corresponds to the A-P axis of the brain.

    use_percentile: the percentile point (from anterior/top)
            to be used as the reference, in the range [0, 100].

    background_dia: the standard deviation (in pixels)
            used to perform background subtraction based on
            a Gaussian filter.

    scale_factor: the brightness-enhancement factor for images
            after background subtraction. (MAY CHANGE IN THE FUTURE)

    alignment_method: currently assumed to be 'ORB'.

    feature_size: the number of feature points to be extracted
            at the maximum. the default value will be used
            if None is specified.

    threshold_factor: a number in the range [0, 1] for filtering
            detected alignments.

    returns
    -------
    matrices: an iterable of (2, 3) affine transform matrices.

    """
    images = tuple(images)
    if len(images) == 1:
        return (_affine.identity(),)
    if alignment_method != 'ORB':
        raise ValueError("only 'ORB' is accepted right now as the alignment method")

    coords = Coordinates.from_image(images[0])
    _perform_alignment = _featurebased.get_alignment_method(
        method=alignment_method,
        feature_size=feature_size,
        scale_factor=scale_factor,
        threshold_factor=threshold_factor,
    )

    points = [_compute.center_of_mass(
        img,
        coords=coords,
        as_int=True
    ) for img in images]

    refidx = _get_reference_index(points, use_percentile=use_percentile)
    ref = images[refidx]

    def _estimate(ref, query):
        ref = _compute.subtract_background(ref, smoothing_dia=background_dia)
        query = _compute.subtract_background(query, smoothing_dia=background_dia)
        alignment = _perform_alignment(ref, query)
        xy_r, xy_q = alignment.as_xy()
        return _affine.estimate(xy_q, xy_r)

    transs = []
    for i in range(len(points)):
        if i == refidx:
            trans = _affine.identity()
        else:
            trans = _estimate(ref, images[i])
        transs.append(trans)
    return tuple(transs)


def _get_reference_index(
    points: Iterable[Point],
    use_percentile: int = 15,
) -> int:
    pos = _np.array([point.i for point in points])
    if pos.size == 2:
        refpos = pos.min()
    else:
        refpos = _math.ceil(_np.percentile(pos, use_percentile))
    return _np.where(pos == refpos)[0][0]
