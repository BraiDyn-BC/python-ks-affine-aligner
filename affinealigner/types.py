from typing import Union, Literal, Iterable
from typing_extensions import Self
from collections import namedtuple as _namedtuple

import numpy as _np
import numpy.typing as _npt

Number = Union[int, float]
AlignmentMethod = Literal['ORB']
ColorSpec = Union[str, Iterable[float]]

# FIXME:
# below I would like to specify the dimensions of the arrays,
# but specifications can lead to an error in a certain case.
# So I leave the dimensions out (resulting in meaningless type specs).
Image = _npt.NDArray
ImageMask = _npt.NDArray[bool]
FloatGrayscaleImage = _npt.NDArray[_np.float32]
Uint8Image = _npt.NDArray[_np.uint8]
RGB24Image = _npt.NDArray[_np.uint8]
AffineMatrix = _npt.NDArray


class Point(_namedtuple('Point', ('i', 'j'))):
    @property
    def x(self) -> Number:
        return self.j

    @property
    def y(self) -> Number:
        return self.i


class Coordinates(_namedtuple('Coordinates', ('I', 'J'))):
    @classmethod
    def create(cls, width: int, height: int) -> Self:
        I, J = _np.meshgrid(_np.arange(height), _np.arange(width), indexing='ij')
        return cls(I=I, J=J)

    @classmethod
    def from_image(cls, img: Image) -> Self:
        H, W = img.shape[:2]
        return cls.create(width=W, height=H)
