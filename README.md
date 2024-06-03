# ks-affine-aligner

An OpenCV-based Python tool for alignment of (presumably clear-skull) images.

## How it works

The `align_image()` method takes a set of images, and works in a following way:

1. **The center of mass (CM)** is computed for each image, and the template (reference)
   image is picked up based on the A-P position of the CM (the actual percentile
   of the reference image among all the input image may be specified using the
   `use_percentile` option).
2. **Gaussian filter-based background subtraction** is performed on each image.
   One can specify the width of the filter using the `background_dia` option (in pixels).
3. **The ORB feature points** are extracted from each image, and alignment is performed
   between the template / reference image (one may be able to tweak this step via
   changing the `feature_size` and `threshold_factor` parameters).
4. **A 2-d affine transformation matrix** is estimated for each image, based on
   the alignment of the feature points.


## Other notes

- The `Coordinates` class represents the IJ coordinate space of the image.
  Pre-computing one may speed up the procedures to a certain extent.
- The routines in the `compute` submodule may be useful for other image-processing purposes.
