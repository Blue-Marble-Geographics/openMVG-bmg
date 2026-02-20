/** @file sift.c
 ** @brief SIFT - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page sift Scale Invariant Feature Transform (SIFT)
@author Andrea Vedaldi
@par "Credits:" May people have contributed with suggestions and bug
reports. Although the following list is certainly incomplete, we would
like to thank: Wei Dong, Loic, Giuseppe, Liu, Erwin, P. Ivanov, and
Q. S. Luo.
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref sift.h implements a @ref sift-usage "SIFT filter object", a
reusable object to extract SIFT features @cite{lowe99object} from one
or multiple images.

- @ref sift-intro
  - @ref sift-intro-detector
  - @ref sift-intro-descriptor
  - @ref sift-intro-extensions
- @ref sift-usage
- @ref sift-tech
  - @ref sift-tech-ss
  - @ref sift-tech-detector
    -  @ref sift-tech-detector-peak
    -  @ref sift-tech-detector-edge
    -  @ref sift-tech-detector-orientation
  - @ref sift-tech-descriptor
    - @ref sift-tech-descriptor-can
    - @ref sift-tech-descriptor-image
    - @ref sift-tech-descriptor-std

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section sift-intro Overview
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A SIFT feature is a selected image region (also called keypoint) with
an associated descriptor. Keypoints are extracted by the <b>@ref
sift-intro-detector "SIFT detector"</b> and their descriptors are
computed by the <b>@ref sift-intro-descriptor "SIFT descriptor"</b>.  It is
also common to use independently the SIFT detector (i.e. computing the
keypoints without descriptors) or the SIFT descriptor (i.e.  computing
descriptors of custom keypoints).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-intro-detector SIFT detector
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@sa
@ref sift-tech-ss "Scale space technical details",
@ref sift-tech-detector "Detector technical details"

A SIFT <em>keypoint</em> is a circular image region with an
orientation. It is described by a geometric <em>frame</em> of four
parameters: the keypoint center coordinates @e x and @e y, its @e
scale (the radius of the region), and its @e orientation (an angle
expressed in radians). The SIFT detector uses as keypoints image
structures which resemble &ldquo;blobs&rdquo;. By searching for blobs
at multiple scales and positions, the SIFT detector is invariant (or,
more accurately, covariant) to translation, rotations, and rescaling
of the image.

The keypoint orientation is also determined from the local image
appearance and is covariant to image rotations. Depending on the
symmetry of the keypoint appearance, determining the orientation can
be ambiguous. In this case, the SIFT detectors returns a list of up to
four possible orientations, constructing up to four frames (differing
only by their orientation) for each detected image blob.

@image html sift-frame.png "SIFT keypoints are circular image regions with an orientation."

There are several parameters that influence the detection of SIFT
keypoints. First, searching keypoints at multiple scales is obtained
by constructing a so-called &ldquo;Gaussian scale space&rdquo;. The
scale space is just a collection of images obtained by progressively
smoothing the input image, which is analogous to gradually reducing
the image resolution. Conventionally, the smoothing level is called
<em>scale</em> of the image. The construction of the scale space is
influenced by the following parameters, set when creating the SIFT
filter object by ::vl_sift_new():

- <b>Number of octaves</b>. Increasing the scale by an octave means
  doubling the size of the smoothing kernel, whose effect is roughly
  equivalent to halving the image resolution. By default, the scale
  space spans as many octaves as possible (i.e. roughly <code>
  log2(min(width,height)</code>), which has the effect of searching
  keypoints of all possible sizes.
- <b>First octave index</b>. By convention, the octave of index 0
  starts with the image full resolution. Specifying an index greater
  than 0 starts the scale space at a lower resolution (e.g. 1 halves
  the resolution). Similarly, specifying a negative index starts the
  scale space at an higher resolution image, and can be useful to
  extract very small features (since this is obtained by interpolating
  the input image, it does not make much sense to go past -1).
- <b>Number of levels per octave</b>. Each octave is sampled at this
  given number of intermediate scales (by default 3). Increasing this
  number might in principle return more refined keypoints, but in
  practice can make their selection unstable due to noise (see [1]).

Keypoints are further refined by eliminating those that are likely to
be unstable, either because they are selected nearby an image edge,
rather than an image blob, or are found on image structures with low
contrast. Filtering is controlled by the follow:

- <b>Peak threshold.</b> This is the minimum amount of contrast to
  accept a keypoint. It is set by configuring the SIFT filter object
  by ::vl_sift_set_peak_thresh().
- <b>Edge threshold.</b> This is the edge rejection threshold. It is
  set by configuring the SIFT filter object by
  ::vl_sift_set_edge_thresh().

<table>
 <caption>Summary of the parameters influencing the SIFT detector.</caption>
 <tr style="font-weight:bold;">
 <td>Parameter</td>
 <td>See also</td>
 <td>Controlled by</td>
 <td>Comment</td>
 </tr>
 <tr>
 <td>number of octaves</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_new</td>
 <td></td>
 </tr>
 <tr>
 <td>first octave index</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_new</td>
 <td>set to -1 to extract very small features</td>
 </tr>
 <tr>
 <td>number of scale levels per octave</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_new</td>
 <td>can affect the number of extracted keypoints</td>
 </tr>
 <tr>
 <td>edge threshold</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_set_edge_thresh</td>
 <td>decrease to eliminate more keypoints</td>
 </tr>
 <tr>
 <td>peak threshold</td>
 <td> @ref sift-intro-detector </td>
 <td>::vl_sift_set_peak_thresh</td>
 <td>increase to eliminate more keypoints</td>
 </tr>
</table>

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-intro-descriptor SIFT Descriptor
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@sa @ref sift-tech-descriptor "Descriptor technical details"

A SIFT descriptor is a 3-D spatial histogram of the image gradients in
characterizing the appearance of a keypoint. The gradient at each
pixel is regarded as a sample of a three-dimensional elementary
feature vector, formed by the pixel location and the gradient
orientation. Samples are weighed by the gradient norm and accumulated
in a 3-D histogram @em h, which (up to normalization and clamping)
forms the SIFT descriptor of the region. An additional Gaussian
weighting function is applied to give less importance to gradients
farther away from the keypoint center. Orientations are quantized into
eight bins and the spatial coordinates into four each, as follows:

@image html sift-descr-easy.png "The SIFT descriptor is a spatial histogram of the image gradient."

SIFT descriptors are computed by either calling
::vl_sift_calc_keypoint_descriptor or
::vl_sift_calc_raw_descriptor. They accept as input a keypoint
frame, which specifies the descriptor center, its size, and its
orientation on the image plane. The following parameters influence the
descriptor calculation:

- <b>magnification factor</b>. The descriptor size is determined by
multiplying the keypoint scale by this factor. It is set by
::vl_sift_set_magnif.
- <b>Gaussian window size</b>. The descriptor support is determined by
a Gaussian window, which discounts gradient contributions farther away
from the descriptor center. The standard deviation of this window is
set by ::vl_sift_set_window_size and expressed in unit of bins.

VLFeat SIFT descriptor uses the following convention. The @em y axis
points downwards and angles are measured clockwise (to be consistent
with the standard image convention). The 3-D histogram (consisting of
@f$ 8 \times 4 \times 4 = 128 @f$ bins) is stacked as a single
128-dimensional vector, where the fastest varying dimension is the
orientation and the slowest the @em y spatial coordinate. This is
illustrated by the following figure.

@image html sift-conv-vlfeat.png "VLFeat conventions"

@note Keypoints (frames) D. Lowe's SIFT implementation convention is
slightly different: The @em y axis points upwards and the angles are
measured counter-clockwise.

@image html sift-conv.png "D. Lowes' SIFT implementation conventions"

<table>
 <caption>Summary of the parameters influencing the SIFT descriptor.</caption>
 <tr style="font-weight:bold;">
 <td>Parameter</td>
 <td>See also</td>
 <td>Controlled by</td>
 <td>Comment</td>
 </tr>
 <tr>
 <td>magnification factor</td>
 <td> @ref sift-intro-descriptor </td>
 <td>::vl_sift_set_magnif</td>
 <td>increase this value to enlarge the image region described</td>
 </tr>
 <tr>
 <td>Gaussian window size</td>
 <td> @ref sift-intro-descriptor </td>
 <td>::vl_sift_set_window_size</td>
 <td>smaller values let the center of the descriptor count more</td>
 </tr>
</table>


<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section sift-intro-extensions Extensions
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<b>Eliminating low-contrast descriptors.</b> Near-uniform patches do
not yield stable keypoints or descriptors. ::vl_sift_set_norm_thresh()
can be used to set a threshold on the average norm of the local
gradient to zero-out descriptors that correspond to very low contrast
regions. By default, the threshold is equal to zero, which means that
no descriptor is zeroed. Normally this option is useful only with
custom keypoints, as detected keypoints are implicitly selected at
high contrast image regions.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section sift-usage Using the SIFT filter object
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The code provided in this module can be used in different ways.  You
can instantiate and use a <b>SIFT filter</b> to extract both SIFT
keypoints and descriptors from one or multiple images. Alternatively,
you can use one of the low level functions to run only a part of the
SIFT algorithm (for instance, to compute the SIFT descriptors of
custom keypoints).

To use a <b>SIFT filter</b> object:

- Initialize a SIFT filter object with ::vl_sift_new(). The filter can
  be reused for multiple images of the same size (e.g. for an entire
  video sequence).
- For each octave in the scale space:
  - Compute the next octave of the DOG scale space using either
   ::vl_sift_process_first_octave() or ::vl_sift_process_next_octave()
   (stop processing if ::VL_ERR_EOF is returned).
  - Run the SIFT detector with ::vl_sift_detect() to get the keypoints.
  - For each keypoint:
    - Use ::vl_sift_calc_keypoint_orientations() to get the keypoint orientation(s).
    - For each orientation:
      - Use ::vl_sift_calc_keypoint_descriptor() to get the keypoint descriptor.
- Delete the SIFT filter by ::vl_sift_delete().

To compute SIFT descriptors of custom keypoints, use
::vl_sift_calc_raw_descriptor().

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section sift-tech Technical details
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-tech-ss Scale space
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

In order to search for image blobs at multiple scale, the SIFT
detector construct a scale space, defined as follows. Let
@f$I_0(\mathbf{x})@f$ denote an idealized <em>infinite resolution</em>
image. Consider the  <em>Gaussian kernel</em>

@f[
 g_{\sigma}(\mathbf{x})
 =
 \frac{1}{2\pi\sigma^2}
 \exp
 \left(
 -\frac{1}{2}
 \frac{\mathbf{x}^\top\mathbf{x}}{\sigma^2}
 \right)
@f]

The <b>Gaussian scale space</b> is the collection of smoothed images

@f[
 I_\sigma = g_\sigma * I,  \quad \sigma \geq 0.
@f]

The image at infinite resolution @f$ I_0 @f$ is useful conceptually,
but is not available to us; instead, the input image @f$ I_{\sigma_n}
@f$ is assumed to be pre-smoothed at a nominal level @f$ \sigma_n =
0.5 @f$ to account for the finite resolution of the pixels. Thus in
practice the scale space is computed by

@f[
I_\sigma = g_{\sqrt{\sigma^2 - \sigma_n^2}} * I_{\sigma_n},
\quad \sigma \geq \sigma_n.
@f]

Scales are sampled at logarithmic steps given by

@f[
\sigma = \sigma_0 2^{o+s/S},
\quad s = 0,\dots,S-1,
\quad o = o_{\min}, \dots, o_{\min}+O-1,
@f]

where @f$ \sigma_0 = 1.6 @f$ is the <em>base scale</em>, @f$ o_{\min}
@f$ is the <em>first octave index</em>, @em O the <em>number of
octaves</em> and @em S the <em>number of scales per octave</em>.

Blobs are detected as local extrema of the <b>Difference of
Gaussians</b> (DoG) scale space, obtained by subtracting successive
scales of the Gaussian scale space:

@f[
\mathrm{DoG}_{\sigma(o,s)} = I_{\sigma(o,s+1)} - I_{\sigma(o,s)}
@f]

At each next octave, the resolution of the images is halved to save
computations. The images composing the Gaussian and DoG scale space
can then be arranged as in the following figure:

@image html sift-ss.png  "GSS and DoG scale space  structures."

The black vertical segments represent images of the Gaussian Scale
Space (GSS), arranged by increasing scale @f$\sigma@f$.  Notice that
the scale level index @e s varies in a slightly redundant set

@f[
s = -1, \dots, S+2
@f]

This simplifies glueing together different octaves and extracting DoG
maxima (required by the SIFT detector).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-tech-detector Detector
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The SIFT frames (keypoints) are extracted based on local extrema
(peaks) of the DoG scale space. Numerically, local extrema are
elements whose @f$ 3 \times 3 \times 3 @f$ neighbors (in space and
scale) have all smaller (or larger) value.  Once extracted, local
extrema are quadratically interpolated (this is very important
especially at the lower resolution scales in order to have accurate
keypoint localization at the full resolution).  Finally, they are
filtered to eliminate low-contrast responses or responses close to
edges and the orientation(s) are assigned, as explained next.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-detector-peak Eliminating low contrast responses
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Peaks which are too short may have been generated by noise and are
discarded.  This is done by comparing the absolute value of the DoG
scale space at the peak with the <b>peak threshold</b> @f$t_p@f$ and
discarding the peak its value is below the threshold.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-detector-edge Eliminating edge responses
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Peaks which are too flat are often generated by edges and do not yield
stable features. These peaks are detected and removed as follows.
Given a peak @f$x,y,\sigma@f$, the algorithm evaluates the @em x,@em y
Hessian of of the DoG scale space at the scale @f$\sigma@f$.  Then the
following score (similar to the Harris function) is computed:

@f[
\frac{(\mathrm{tr}\,D(x,y,\sigma))^2}{\det D(x,y,\sigma)},
\quad
D =
\left[
\begin{array}{cc}
\frac{\partial^2 \mathrm{DoG}}{\partial x^2} &
\frac{\partial^2 \mathrm{DoG}}{\partial x\partial y} \\
\frac{\partial^2 \mathrm{DoG}}{\partial x\partial y} &
\frac{\partial^2 \mathrm{DoG}}{\partial y^2}
\end{array}
\right].
 @f]

This score has a minimum (equal to 4) when both eigenvalues of the
Jacobian are equal (curved peak) and increases as one of the
eigenvalues grows and the other stays small. Peaks are retained if the
score is below the quantity @f$(t_e+1)(t_e+1)/t_e@f$, where @f$t_e@f$
is the <b>edge threshold</b>. Notice that this quantity has a minimum
equal to 4 when @f$t_e=1@f$ and grows thereafter. Therefore the range
of the edge threshold is @f$[1,\infty)@f$.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-tech-detector-orientation Orientation assignment
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A peak in the DoG scale space fixes 2 parameters of the keypoint: the
position and scale. It remains to choose an orientation. In order to
do this, SIFT computes an histogram of the gradient orientations in a
Gaussian window with a standard deviation which is 1.5 times bigger
than the scale @f$\sigma@f$ of the keypoint.

@image html sift-orient.png

This histogram is then smoothed and the maximum is selected.  In
addition to the biggest mode, up to other three modes whose amplitude
is within the 80% of the biggest mode are retained and returned as
additional orientations.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection sift-tech-descriptor Descriptor
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A SIFT descriptor of a local region (keypoint) is a 3-D spatial
histogram of the image gradients.  The gradient at each pixel is
regarded as a sample of a three-dimensional elementary feature vector,
formed by the pixel location and the gradient orientation. Samples are
weighed by the gradient norm and accumulated in a 3-D histogram @em h,
which (up to normalization and clamping) forms the SIFT descriptor of
the region. An additional Gaussian weighting function is applied to
give less importance to gradients farther away from the keypoint
center.

<!-- @image html sift-bins.png -->

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-descriptor-can Construction in the canonical frame
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Denote the gradient vector field computed at the scale @f$ \sigma @f$ by
@f[
  J(x,y) = \nalba I_\sigma(x,y)
  =
  \left[\begin{array}{cc}
  \frac{\partial I_\sigma}{\partial x} &
  \frac{\partial I_\sigma}{\partial y} &
  \end{array}\right]
@f]

The descriptor is a 3-D spatial histogram capturing the distribution
of @f$ J(x,y) @f$. It is convenient to describe its construction in
the <em>canonical frame</em>. In this frame, the image and descriptor
axes coincide and each spatial bin has side 1. The histogram has @f$
N_\theta \times N_x \times N_y @f$ bins (usually @f$ 8 \times 4 \times
4 @f$), as in the following figure:

@image html sift-can.png Canonical SIFT descriptor and spatial binning functions

Bins are indexed by a triplet of indexes <em>t, i, j</em> and their
centers are given by

@f{eqnarray*}
 \theta_t &=& \frac{2\pi}{N_\theta} t, \quad t = 0,\dots,N_{\theta}-1, \\
 x_i &=& i - \frac{N_x -1}{2}, \quad i = 0,\dots,N_x-1, \\
 y_j &=& j - \frac{N_x -1}{2}, \quad j = 0,\dots,N_y-1. \\
@f}

The histogram is computed by using trilinear interpolation, i.e.  by
weighing contributions by the <em>binning functions</em>

@f{eqnarray*}
  \displaystyle
  w(z) &=& \mathrm{max}(0, 1 - |z|),
  \\
  \displaystyle
  w_\mathrm{ang}(z) &=& \sum_{k=-\infty}^{+\infty}
  w\left(
  \frac{N_\theta}{2\pi} z + N_\theta k
  \right).
@f}

The gradient vector field is transformed in a three-dimensional
density map of weighed contributions

@f[
   f(\theta, x, y) = |J(x,y)| \delta(\theta - \angle J(x,y))
@f]

The historam is localized in the keypoint support by a Gaussian window
of standard deviation @f$ \sigma_{\mathrm{win}} @f$. The histogram is
then given by

@f{eqnarray*}
 h(t,i,j) &=& \int
 g_{\sigma_\mathrm{win}}(x,y)
 w_\mathrm{ang}(\theta - \theta_t) w(x-x_i) w(y-y_j)
 f(\theta,x,y)
 d\theta\,dx\,dy
\\
&=& \int
 g_{\sigma_\mathrm{win}}(x,y)
 w_\mathrm{ang}(\angle J(x,y) - \theta_t) w(x-x_i) w(y-y_j)
 |J(x,y)|\,dx\,dy
@f}

In post processing, the histogram is @f$ l^2 @f$ normalized, then
clamped at 0.2, and @f$ l^2 @f$ normalized again.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-descriptor-image Calculation in the image frame
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Invariance to similarity transformation is attained by attaching
descriptors to SIFT keypoints (or other similarity-covariant frames).
Then projecting the image in the canonical descriptor frames has the
effect of undoing the image deformation.

In practice, however, it is convenient to compute the descriptor
directly in the image frame. To do this, denote with a hat quantities
relative to the canonical frame and without a hat quantities relative
to the image frame (so for instance @f$ \hat x @f$ is the @e
x-coordinate in the canonical frame and @f$ x @f$ the x-coordinate in
the image frame). Assume that canonical and image frame are
related by an affinity:

@f[
  \mathbf{x} = A \hat\mathbf{x} + T,
  \qquad
  \mathbf{x} =
  \left[\begin{array}{cc}
    x \\
    y
  \end{arraty}\right],
  \quad
  \hat\mathbf{x} =
  \left[\begin{array}{cc}
    \hat x \\
    \hat y
  \end{arraty}\right].
@f]

@image html sift-image-frame.png

Then all quantities can be computed in the image frame directly. For instance,
the image at infinite resolution in the two frames are related by

@f[
 \hat I_0(\hat{\mathbf{x}})  = I_0(\mathbf{x}),
 \qquad
 \mathbf{x} = A \hat{\mathbf{x}} + T.
@f]

The canonized image at scale @f$ \hat \sigma @f$ is in relation with the scaled image

@f[
 \hat I_{\hat{\sigma}}(\hat{\mathbf{x}})  = I_{A\hat{\sigma}}(\mathbf{x}),
 \qquad \mathbf{x} = A \hat{\mathbf{x}} + T
@f]

where, by generalizing the previous definitions, we have

@f[
 I_{A\hat \sigma}(\mathbf{x}) = (g_{A\hat\sigma} * I_0)(\mathbf{x}),
\quad
 g_{A\hat\sigma}(\mathbf{x})
 =
 \frac{1}{2\pi|A|\hat \sigma^2}
 \exp
 \left(
 -\frac{1}{2}
 \frac{\mathbf{x}^\top A^{-\top}A^{-1}\mathbf{x}}{\hat \sigma^2}
 \right)
@f]

Deriving shows that the gradient fields are in relation

@f[
  \hat J(\hat{\mathbf{x}}) = J(\mathbf{x}) A,
 \quad J(\mathbf{x}) = (\nabla I_{A\hat\sigma})(\mathbf{x}),
 \qquad \mathbf{x} = A \hat{\mathbf{x}} + T.
@f]

Therefore we can compute the descriptor either in the image or canonical frame as:

@f{eqnarray*}
 h(t,i,j)
 &=&
 \int
 g_{\hat \sigma_\mathrm{win}}(\hat{\mathbf{x}})\,
 w_\mathrm{ang}(\angle \hat J(\hat{\mathbf{x}}) - \theta_t)\,
 w_{ij}(\hat{\mathbf{x}})\,
 |\hat J(\hat{\mathbf{x}})|\,
 d\hat{\mathbf{x}}
 \\
 &=& \int
 g_{A \hat \sigma_\mathrm{win}}(\mathbf{x} - T)\,
 w_\mathrm{ang}(\angle J(\mathbf{x})A - \theta_t)\,
 w_{ij}(A^{-1}(\mathbf{x} - T))\,
 |J(\mathbf{x})A|\,
 d\mathbf{x}.
@f}

where we defined the product of the two spatial binning functions

@f[
 w_{ij}(\hat{\mathbf{x}}) = w(\hat x - \hat x_i) w(\hat y - \hat y_j)
@f]


In the actual implementation, this integral is computed by visiting a
rectangular area of the image that fully contains the keypoint grid
(along with half a bin border to fully include the bin windowing
function). Since the descriptor can be rotated, this area is a
rectangle of sides @f$m/2\sqrt{2} (N_x+1,N_y+1)@f$ (see also the
illustration).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsubsection sift-tech-descriptor-std Standard SIFT descriptor
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

For a SIFT-detected keypoint of center @f$ T @f$, scale @f$ \sigma @f$
and orientation @f$ \theta @f$, the affine transformation @f$ (A,T)
@f$ reduces to the similarity transformation

@f[
     \mathbf{x} = m \sigma R(\theta) \hat{\mathbf{x}} + T
@f]

where @f$ R(\theta) @f$ is a counter-clockwise rotation of @f$ \theta
@f$ radians, @f$ m \mathcal{\sigma} @f$ is the size of a descriptor
bin in pixels, and @e m is the <b>descriptor magnification factor</b>
which expresses how much larger a descriptor bin is compared to
the scale of the keypoint @f$ \sigma @f$
(the default value is @e m = 3). Moreover, the
standard SIFT descriptor computes the image gradient at the scale of
the keypoints, which in the canonical frame is equivalent to a
smoothing of @f$ \hat \sigma = 1/m @f$. Finally, the default
Gaussian window size is set to have standard deviation
 @f$ \hat \sigma_\mathrm{win} = 2 @f$. This yields the formula

@f{eqnarray*}
 h(t,i,j)
 &=&
 m \sigma \int
 g_{\sigma_\mathrm{win}}(\mathbf{x} - T)\,
 w_\mathrm{ang}(\angle J(\mathbf{x}) - \theta  - \theta_t)\,
 w_{ij}\left(\frac{R(\theta)^\top \mathbf{x} - T}{m\sigma}\right)\,
 |J(\mathbf{x})|\,
 d\mathbf{x},
\\
\sigma_{\mathrm{win}} &=& m\sigma\hat \sigma_{\mathrm{win}},
\\
 J(\mathbf{x})
 &=& \nabla (g_{m \sigma \hat \sigma} * I)(\mathbf{x})
 = \nabla (g_{\sigma} * I)(\mathbf{x})
 = \nabla I_{\sigma} (\mathbf{x}).
@f}



**/

#include "../../../P2PUtils.h"

#include "sift.h"
#include "imopv.h"
#include "mathop.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/** @internal @brief Use bilinear interpolation to compute orientations */
#define VL_SIFT_BILINEAR_ORIENTATIONS 1

#define EXPN_MAX 25.0         /**< ::fast_expn table max  @internal */

#define NBO 8
#define NBP 4

#define log2(x) (log(x)/VL_LOG_OF_2)

#define REDUCE_MEMORY /* Share buffer space to reduce memory use 25% */

extern int hasAVX2;

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Fast @f$exp(-x)@f$ approximation
 **
 ** @param x argument.
 **
 ** The argument must be in the range [0, ::EXPN_MAX] .
 **
 ** @return approximation of @f$exp(-x)@f$.
 **/

VL_INLINE double
fast_expn (VlSiftFilt const * filter, double x)
{
  double a,b,r ;
  int i ;
  /*assert(0 <= x && x <= EXPN_MAX) ;*/

  if (x > EXPN_MAX) return 0.0 ;

  x *= EXPN_SZ / EXPN_MAX ;
  i = (int)vl_floor_d (x) ;
  r = x - i ;
  a = filter->expn_tab [i    ] ;
  b = filter->expn_tab [i + 1] ;
  return a + r * (b - a) ;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Initialize tables for ::fast_expn
 **/

VL_INLINE void
fast_expn_init (VlSiftFilt * filter)
{
  int k  ;
  for(k = 0 ; k < EXPN_SZ + 1 ; ++ k) {
    filter->expn_tab [k] = exp (- (double) k * (EXPN_MAX / EXPN_SZ)) ;
  }
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy image, upsample rows and take transpose
 **
 ** @param dst     output image buffer.
 ** @param src     input image buffer.
 ** @param width   input image width.
 ** @param height  input image height.
 **
 ** The output image has dimensions @a height by 2 @a width (so the
 ** destination buffer must be at least as big as two times the
 ** input buffer).
 **
 ** Upsampling is performed by linear interpolation.
 **/

static void
copy_and_upsample_rows
(vl_sift_pix       *dst,
 vl_sift_pix const *src, int width, int height)
{
  int x, y ;
  vl_sift_pix a, b ;

  for(y = 0 ; y < height ; ++y) {
    b = a = *src++ ;
    for(x = 0 ; x < width - 1 ; ++x) {
      b = *src++ ;
      *dst = a ;             dst += height ;
      *dst = 0.5 * (a + b) ; dst += height ;
      a = b ;
    }
    *dst = b ; dst += height ;
    *dst = b ; dst += height ;
    dst += 1 - width * 2 * height ;
  }
}

#if defined(_MSC_VER)
#define VL_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define VL_RESTRICT __restrict__
#else
#define VL_RESTRICT
#endif

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Smooth an image
 ** @param self        SIFT filter.
 ** @param outputImage output imgae buffer.
 ** @param tempImage   temporary image buffer.
 ** @param inputImage  input image buffer.
 ** @param width       input image width.
 ** @param height      input image height.
 ** @param sigma       smoothing.
 **/

#if 1
void
GaussianRowSymmetricClampSSE2(
  float* VL_RESTRICT dst,
  const float* VL_RESTRICT src,
  size_t width,
  const float* VL_RESTRICT k,
  int W)
{
  size_t x = 0;

  /* left edge */
  for (; x < (size_t)W && x < width; ++x) {
    float acc = k[0] * src[x];
    int i;
    for (i = 1; i <= W; ++i) {
      size_t xm = (x < (size_t)i) ? 0 : x - (size_t)i;
      size_t xp = (x + (size_t)i >= width) ? width - 1 : x + (size_t)i;
      acc += k[i] * (src[xm] + src[xp]);
    }
    dst[x] = acc;
  }

  // Nothing to vectorize if width <= W
  if (width <= (size_t)W) {
    return;
  }

  /* center */
  for (; x + 3 < width - (size_t)W; x += 4) {
    __m128 acc = _mm_mul_ps(_mm_set1_ps(k[0]),
      _mm_loadu_ps(src + x));

    int i;
    for (i = 1; i <= W; ++i) {
      __m128 km = _mm_set1_ps(k[i]);
      __m128 a = _mm_loadu_ps(src + x - i);
      __m128 b = _mm_loadu_ps(src + x + i);
      acc = _mm_add_ps(acc, _mm_mul_ps(km, _mm_add_ps(a, b)));
    }

    _mm_storeu_ps(dst + x, acc);
  }

  /* right edge */
  for (; x < width; ++x) {
    float acc = k[0] * src[x];
    int i;
    for (i = 1; i <= W; ++i) {
      size_t xm = (x < (size_t)i) ? 0 : x - (size_t)i;
      size_t xp = (x + (size_t)i >= width) ? width - 1 : x + (size_t)i;
      acc += k[i] * (src[xm] + src[xp]);
    }
    dst[x] = acc;
  }
}

void
GaussianRowSymmetricClampAVX2(
  float* VL_RESTRICT dst,
  const float* VL_RESTRICT src,
  size_t width,
  const float* VL_RESTRICT k,
  int W);

void AccumulateW13_AVX2_Block(
  float k0,
  const float* VL_RESTRICT rowCenter,
  float* const VL_RESTRICT rowUp[64],
  float* const VL_RESTRICT rowDn[64],
  const float* VL_RESTRICT kw,
  float* VL_RESTRICT out,
  int width);

typedef void (*GaussianRowFn)(
  float*,
  const float*,
  vl_size,
  const float*,
  int);

static GaussianRowFn gGaussianRowFn = 0;

static void
_vl_sift_smooth(
  VlSiftFilt* self,
  vl_sift_pix* VL_RESTRICT outputImage,
  vl_sift_pix* VL_RESTRICT tempImage, /* unused, ABI only */
  vl_sift_pix const* VL_RESTRICT inputImage,
  vl_size width,
  vl_size height,
  double sigma)
{
  if (hasAVX2) {
    gGaussianRowFn = GaussianRowSymmetricClampAVX2;
  }
  else {
    gGaussianRowFn = GaussianRowSymmetricClampSSE2;
  }

  (void)tempImage;

  /* ------------------------------------------------------------
   * Build symmetric Gaussian kernel (half only)
   * ------------------------------------------------------------ */
  if (self->gaussFilterSigma != sigma) {
    int i;
    float acc = 0.0f;

    self->gaussFilterWidth = VL_MAX((int)ceil(4.0 * sigma), 1);
    {
      int W = self->gaussFilterWidth;
      size_t needed = sizeof(vl_sift_pix) * (size_t)(W + 1);

      if (needed > self->gaussFilterSize) {
        if (self->gaussFilter) vl_free(self->gaussFilter);
        self->gaussFilter = vl_malloc(needed);
        self->gaussFilterSize = needed;
      }

      self->gaussFilterSigma = sigma;

      for (i = 0; i <= W; ++i) {
        float d = (float)i / (float)sigma;
        float v = expf(-0.5f * d * d);
        self->gaussFilter[i] = v;
        acc += (i == 0) ? v : 2.0f * v;
      }

      for (i = 0; i <= W; ++i) {
        self->gaussFilter[i] /= acc;
      }
    }
  }

  {
    const int W = self->gaussFilterWidth;
    const float* VL_RESTRICT k = self->gaussFilter;

    if (W == 0) {
      memcpy(outputImage, inputImage,
        sizeof(vl_sift_pix) * width * height);
      return;
    }

    /* ------------------------------------------------------------
     * Ring buffer: exactly 2W+1 horizontally filtered rows
     * ------------------------------------------------------------ */
    {
      const int R = 2 * W + 1;
      float* VL_RESTRICT rowBuf =
        (float*)vl_malloc(sizeof(float) * (size_t)R * width);

      const int Wplus1 = W + 1;
      int center = W;
      int t;

      GaussianRowFn rowFn = gGaussianRowFn;
      float* VL_RESTRICT rowPtr[64];
      int i;
      for (i = 0; i < R; ++i)
        rowPtr[i] = rowBuf + (size_t)i * width;

      /* ----------------------------------------------------------
       * Initialize window for y = 0 (rows [-W .. +W] clamped)
       * ---------------------------------------------------------- */
      for (t = -W; t <= W; ++t) {
        vl_size srcY;
        if (t < 0) {
          srcY = 0;
        }
        else {
          srcY = (vl_size)t;
          if (srcY >= height) srcY = height - 1;
        }

        rowFn(
          rowBuf + (size_t)(t + W) * width,
          inputImage + srcY * width,
          width,
          k,
          W);
      }

      /* Helper to slide the vertical window by one row. */
#define SLIDE_RING() do {                                  \
  int add = center + Wplus1;                                \
  if (add >= R) add -= R;                                   \
                                                            \
  vl_size srcY = y + (vl_size)Wplus1;                       \
  if (srcY >= height) srcY = height - 1;                    \
                                                            \
  rowFn(rowPtr[add],                                        \
        inputImage + srcY * width,                          \
        width, k, W);                                       \
                                                            \
  center++;                                                 \
  if (center == R) center = 0;                              \
} while(0)

    /* ----------------------------------------------------------
     * Main loop (W-dispatch hoisted, MSVC optimized)
     * ---------------------------------------------------------- */
      {
        vl_size y, x;
        int i;

        /* Hoist kernel coefficients ONCE */
        float k0 = k[0];
        float kw[64];
        for (i = 1; i <= W; ++i)
          kw[i] = k[i];

        float* VL_RESTRICT rowUp[64];
        float* VL_RESTRICT rowDn[64];

        switch (W) {

          /* ======================================================
           * W == 5
           * ====================================================== */
        case 5:
          for (y = 0; y < height; ++y) {

            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= 5; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x + 3 < width; x += 4) {

              float a0 =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]);

              float a1 =
                k0 * rowCenter[x + 1] +
                kw[1] * (rowUp[1][x + 1] + rowDn[1][x + 1]) +
                kw[2] * (rowUp[2][x + 1] + rowDn[2][x + 1]) +
                kw[3] * (rowUp[3][x + 1] + rowDn[3][x + 1]) +
                kw[4] * (rowUp[4][x + 1] + rowDn[4][x + 1]) +
                kw[5] * (rowUp[5][x + 1] + rowDn[5][x + 1]);

              float a2 =
                k0 * rowCenter[x + 2] +
                kw[1] * (rowUp[1][x + 2] + rowDn[1][x + 2]) +
                kw[2] * (rowUp[2][x + 2] + rowDn[2][x + 2]) +
                kw[3] * (rowUp[3][x + 2] + rowDn[3][x + 2]) +
                kw[4] * (rowUp[4][x + 2] + rowDn[4][x + 2]) +
                kw[5] * (rowUp[5][x + 2] + rowDn[5][x + 2]);

              float a3 =
                k0 * rowCenter[x + 3] +
                kw[1] * (rowUp[1][x + 3] + rowDn[1][x + 3]) +
                kw[2] * (rowUp[2][x + 3] + rowDn[2][x + 3]) +
                kw[3] * (rowUp[3][x + 3] + rowDn[3][x + 3]) +
                kw[4] * (rowUp[4][x + 3] + rowDn[4][x + 3]) +
                kw[5] * (rowUp[5][x + 3] + rowDn[5][x + 3]);

              out[x] = a0;
              out[x + 1] = a1;
              out[x + 2] = a2;
              out[x + 3] = a3;
            }

            for (; x < width; ++x) {
              out[x] =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]);
            }

            SLIDE_RING();
          }
          break;

          /* ======================================================
           * W == 7
           * ====================================================== */
        case 7:
          for (y = 0; y < height; ++y) {

            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= 7; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x + 3 < width; x += 4) {
              float a0 =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]) +
                kw[6] * (rowUp[6][x] + rowDn[6][x]) +
                kw[7] * (rowUp[7][x] + rowDn[7][x]);

              float a1 =
                k0 * rowCenter[x + 1] +
                kw[1] * (rowUp[1][x + 1] + rowDn[1][x + 1]) +
                kw[2] * (rowUp[2][x + 1] + rowDn[2][x + 1]) +
                kw[3] * (rowUp[3][x + 1] + rowDn[3][x + 1]) +
                kw[4] * (rowUp[4][x + 1] + rowDn[4][x + 1]) +
                kw[5] * (rowUp[5][x + 1] + rowDn[5][x + 1]) +
                kw[6] * (rowUp[6][x + 1] + rowDn[6][x + 1]) +
                kw[7] * (rowUp[7][x + 1] + rowDn[7][x + 1]);

              float a2 =
                k0 * rowCenter[x + 2] +
                kw[1] * (rowUp[1][x + 2] + rowDn[1][x + 2]) +
                kw[2] * (rowUp[2][x + 2] + rowDn[2][x + 2]) +
                kw[3] * (rowUp[3][x + 2] + rowDn[3][x + 2]) +
                kw[4] * (rowUp[4][x + 2] + rowDn[4][x + 2]) +
                kw[5] * (rowUp[5][x + 2] + rowDn[5][x + 2]) +
                kw[6] * (rowUp[6][x + 2] + rowDn[6][x + 2]) +
                kw[7] * (rowUp[7][x + 2] + rowDn[7][x + 2]);

              float a3 =
                k0 * rowCenter[x + 3] +
                kw[1] * (rowUp[1][x + 3] + rowDn[1][x + 3]) +
                kw[2] * (rowUp[2][x + 3] + rowDn[2][x + 3]) +
                kw[3] * (rowUp[3][x + 3] + rowDn[3][x + 3]) +
                kw[4] * (rowUp[4][x + 3] + rowDn[4][x + 3]) +
                kw[5] * (rowUp[5][x + 3] + rowDn[5][x + 3]) +
                kw[6] * (rowUp[6][x + 3] + rowDn[6][x + 3]) +
                kw[7] * (rowUp[7][x + 3] + rowDn[7][x + 3]);

              out[x] = a0;
              out[x + 1] = a1;
              out[x + 2] = a2;
              out[x + 3] = a3;
            }

            for (; x < width; ++x) {
              out[x] =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]) +
                kw[6] * (rowUp[6][x] + rowDn[6][x]) +
                kw[7] * (rowUp[7][x] + rowDn[7][x]);
            }

            SLIDE_RING();
          }
          break;

        case 8:
          for (y = 0; y < height; ++y) {

            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= 8; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x + 3 < width; x += 4) {

              float a0 =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]) +
                kw[6] * (rowUp[6][x] + rowDn[6][x]) +
                kw[7] * (rowUp[7][x] + rowDn[7][x]) +
                kw[8] * (rowUp[8][x] + rowDn[8][x]);

              float a1 =
                k0 * rowCenter[x + 1] +
                kw[1] * (rowUp[1][x + 1] + rowDn[1][x + 1]) +
                kw[2] * (rowUp[2][x + 1] + rowDn[2][x + 1]) +
                kw[3] * (rowUp[3][x + 1] + rowDn[3][x + 1]) +
                kw[4] * (rowUp[4][x + 1] + rowDn[4][x + 1]) +
                kw[5] * (rowUp[5][x + 1] + rowDn[5][x + 1]) +
                kw[6] * (rowUp[6][x + 1] + rowDn[6][x + 1]) +
                kw[7] * (rowUp[7][x + 1] + rowDn[7][x + 1]) +
                kw[8] * (rowUp[8][x + 1] + rowDn[8][x + 1]);

              float a2 =
                k0 * rowCenter[x + 2] +
                kw[1] * (rowUp[1][x + 2] + rowDn[1][x + 2]) +
                kw[2] * (rowUp[2][x + 2] + rowDn[2][x + 2]) +
                kw[3] * (rowUp[3][x + 2] + rowDn[3][x + 2]) +
                kw[4] * (rowUp[4][x + 2] + rowDn[4][x + 2]) +
                kw[5] * (rowUp[5][x + 2] + rowDn[5][x + 2]) +
                kw[6] * (rowUp[6][x + 2] + rowDn[6][x + 2]) +
                kw[7] * (rowUp[7][x + 2] + rowDn[7][x + 2]) +
                kw[8] * (rowUp[8][x + 2] + rowDn[8][x + 2]);

              float a3 =
                k0 * rowCenter[x + 3] +
                kw[1] * (rowUp[1][x + 3] + rowDn[1][x + 3]) +
                kw[2] * (rowUp[2][x + 3] + rowDn[2][x + 3]) +
                kw[3] * (rowUp[3][x + 3] + rowDn[3][x + 3]) +
                kw[4] * (rowUp[4][x + 3] + rowDn[4][x + 3]) +
                kw[5] * (rowUp[5][x + 3] + rowDn[5][x + 3]) +
                kw[6] * (rowUp[6][x + 3] + rowDn[6][x + 3]) +
                kw[7] * (rowUp[7][x + 3] + rowDn[7][x + 3]) +
                kw[8] * (rowUp[8][x + 3] + rowDn[8][x + 3]);

              out[x] = a0;
              out[x + 1] = a1;
              out[x + 2] = a2;
              out[x + 3] = a3;
            }

            for (; x < width; ++x) {
              out[x] =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]) +
                kw[6] * (rowUp[6][x] + rowDn[6][x]) +
                kw[7] * (rowUp[7][x] + rowDn[7][x]) +
                kw[8] * (rowUp[8][x] + rowDn[8][x]);
            }

            SLIDE_RING();
          }
          break;

          /* ======================================================
           * W == 10 (2-accumulator scalar)
           * ====================================================== */
        case 10:
          for (y = 0; y < height; ++y) {
            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= 10; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x < width; ++x) {

              float acc0 = 0.0f;
              float acc1 = 0.0f;

              acc0 += kw[1] * (rowUp[1][x] + rowDn[1][x]);
              acc1 += kw[2] * (rowUp[2][x] + rowDn[2][x]);
              acc0 += kw[3] * (rowUp[3][x] + rowDn[3][x]);
              acc1 += kw[4] * (rowUp[4][x] + rowDn[4][x]);
              acc0 += kw[5] * (rowUp[5][x] + rowDn[5][x]);
              acc1 += kw[6] * (rowUp[6][x] + rowDn[6][x]);
              acc0 += kw[7] * (rowUp[7][x] + rowDn[7][x]);
              acc1 += kw[8] * (rowUp[8][x] + rowDn[8][x]);
              acc0 += kw[9] * (rowUp[9][x] + rowDn[9][x]);
              acc1 += kw[10] * (rowUp[10][x] + rowDn[10][x]);

              out[x] = k0 * rowCenter[x] + acc0 + acc1;
            }

            SLIDE_RING();
          }
          break;

          /* ======================================================
           * W == 13 (AVX2 fast path)
           * ====================================================== */
        case 13:
          if (hasAVX2) {
            for (y = 0; y < height; ++y) {

              float* VL_RESTRICT out = outputImage + y * width;
              float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

              for (i = 1; i <= 13; ++i) {
                int up = center - i;
                int dn = center + i;
                if (up < 0) up += R;
                if (dn >= R) dn -= R;
                rowUp[i] = rowBuf + (size_t)up * width;
                rowDn[i] = rowBuf + (size_t)dn * width;
              }

              AccumulateW13_AVX2_Block(k0, rowCenter, rowUp, rowDn, kw, out, width);

              SLIDE_RING();
            }
          }
          else {
            for (y = 0; y < height; ++y) {
              /* scalar fallback */

              float* VL_RESTRICT out = outputImage + y * width;
              float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

              for (i = 1; i <= 13; ++i) {
                int up = center - i;
                int dn = center + i;
                if (up < 0) up += R;
                if (dn >= R) dn -= R;
                rowUp[i] = rowBuf + (size_t)up * width;
                rowDn[i] = rowBuf + (size_t)dn * width;
              }
              for (x = 0; x < width; ++x) {
                float acc = k0 * rowCenter[x];
                for (i = 1; i <= 13; ++i)
                  acc += kw[i] * (rowUp[i][x] + rowDn[i][x]);
                out[x] = acc;
              }

              SLIDE_RING();
            }
          }
          break;

          /* ======================================================
           * Fallback
           * ====================================================== */
        default:
          for (y = 0; y < height; ++y) {

            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= W; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x + 1 < width; x += 2) {

              float a0 = k0 * rowCenter[x];
              float a1 = k0 * rowCenter[x + 1];

              for (i = 1; i <= W; ++i) {
                a0 += kw[i] * (rowUp[i][x] + rowDn[i][x]);
                a1 += kw[i] * (rowUp[i][x + 1] + rowDn[i][x + 1]);
              }

              out[x] = a0;
              out[x + 1] = a1;
            }

            for (; x < width; ++x) {
              float acc = k0 * rowCenter[x];
              for (i = 1; i <= W; ++i)
                acc += kw[i] * (rowUp[i][x] + rowDn[i][x]);
              out[x] = acc;
            }

            SLIDE_RING();
          }
          break;
        }
      }

      vl_free(rowBuf);
    }
  }
}

#else
static void
_vl_sift_smooth (VlSiftFilt * self,
                 vl_sift_pix * outputImage,
                 vl_sift_pix * tempImage,
                 vl_sift_pix const * inputImage,
                 vl_size width,
                 vl_size height,
                 double sigma)
{
  /* prepare Gaussian filter */
  if (self->gaussFilterSigma != sigma) {
    vl_uindex j ;
    vl_sift_pix acc = 0 ;

    self->gaussFilterWidth = VL_MAX(ceil(4.0 * sigma), 1) ;
    size_t const gaussFilterSize = sizeof(vl_sift_pix) * (2 * self->gaussFilterWidth + 1);
    if (gaussFilterSize > self->gaussFilterSize) {
    if (self->gaussFilter) vl_free (self->gaussFilter) ;
        self->gaussFilter = vl_malloc (gaussFilterSize) ;
        self->gaussFilterSize = gaussFilterSize;
    }

    self->gaussFilterSigma = sigma ;

    for (j = 0 ; j < 2 * self->gaussFilterWidth + 1 ; ++j) {
      vl_sift_pix d = ((vl_sift_pix)((signed)j - (signed)self->gaussFilterWidth)) / ((vl_sift_pix)sigma) ;
      self->gaussFilter[j] = (vl_sift_pix) exp (- 0.5 * (d*d)) ;
      acc += self->gaussFilter[j] ;
    }
    for (j = 0 ; j < 2 * self->gaussFilterWidth + 1 ; ++j) {
      self->gaussFilter[j] /= acc ;
    }
  }

  if (self->gaussFilterWidth == 0) {
    memcpy (outputImage, inputImage, sizeof(vl_sift_pix) * width * height) ;
    return ;
  }

  vl_imconvcol_vf (tempImage, height,
                   inputImage, width, height, width,
                   self->gaussFilter,
                   - self->gaussFilterWidth, self->gaussFilterWidth,
                   1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;

  vl_imconvcol_vf (outputImage, width,
                   tempImage, height, width, height,
                   self->gaussFilter,
                   - self->gaussFilterWidth, self->gaussFilterWidth,
                   1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;
}
#endif

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy and downsample an image
 **
 ** @param dst    output imgae buffer.
 ** @param src    input  image buffer.
 ** @param width  input  image width.
 ** @param height input  image height.
 ** @param d      octaves (non negative).
 **
 ** The function downsamples the image @a d times, reducing it to @c
 ** 1/2^d of its original size. The parameters @a width and @a height
 ** are the size of the input image. The destination image @a dst is
 ** assumed to be <code>floor(width/2^d)</code> pixels wide and
 ** <code>floor(height/2^d)</code> pixels high.
 **/

static void
copy_and_downsample
(vl_sift_pix       *dst,
 vl_sift_pix const *src,
 int width, int height, int d)
{
  int x, y ;

  d = 1 << d ; /* d = 2^d */
  for(y = 0 ; y < height ; y+=d) {
    vl_sift_pix const * srcrowp = src + y * width ;
    for(x = 0 ; x < width - (d-1) ; x+=d) {
      *dst++ = *srcrowp ;
      srcrowp += d ;
    }
  }
}

/** ------------------------------------------------------------------
 ** @brief Create a new SIFT filter
 **
 ** @param width    image width.
 ** @param height   image height.
 ** @param noctaves number of octaves.
 ** @param nlevels  number of levels per octave.
 ** @param o_min    first octave index.
 **
 ** The function allocates and returns a new SIFT filter for the
 ** specified image and scale space geometry.
 **
 ** Setting @a O to a negative value sets the number of octaves to the
 ** maximum possible value depending on the size of the image.
 **
 ** @return the new SIFT filter.
 ** @sa ::vl_sift_delete().
 **/

VL_EXPORT
VlSiftFilt *
vl_sift_new (int width, int height,
             int noctaves, int nlevels,
             int o_min)
{
  VlSiftFilt *f = vl_malloc (sizeof(VlSiftFilt)) ;

  int w   = VL_SHIFT_LEFT (width,  -o_min) ;
  int h   = VL_SHIFT_LEFT (height, -o_min) ;
  int nel = w * h ;

  /* negative value O => calculate max. value */
  if (noctaves < 0) {
    noctaves = VL_MAX (floor (log2 (VL_MIN(width, height))) - o_min - 3, 1) ;
  }

  f-> width   = width ;
  f-> height  = height ;
  f-> O       = noctaves ;
  f-> S       = nlevels ;
  f-> o_min   = o_min ;
  f-> s_min   = -1 ;
  f-> s_max   = nlevels + 1 ;
  f-> o_cur   = o_min ;

#ifdef REDUCE_MEMORY
  const size_t commonSize =
    sizeof(vl_sift_pix) *
    max(
          nel * ( f->s_max - f->s_min ), /* dog */
          nel * 2 * ( f->s_max - f->s_min ) /* grad */
    );
  f-> dog     = vl_malloc(commonSize) ;
  f-> grad    = f->dog ;

  f-> octave  = vl_malloc (sizeof(vl_sift_pix) * nel
                            * (f->s_max - f->s_min + 1)  ) ;
  /* Dog and temp used together, so temp must follow dog. */
  f-> temp = f->dog + nel * ( f->s_max - f->s_min ) ;
#else
  f-> temp    = vl_malloc (sizeof(vl_sift_pix) * nel    ) ;
  f-> octave  = vl_malloc (sizeof(vl_sift_pix) * nel
                        * (f->s_max - f->s_min + 1)  ) ;
  f-> dog     = vl_malloc (sizeof(vl_sift_pix) * nel
                        * (f->s_max - f->s_min    )  ) ;
  f-> grad    = vl_malloc (sizeof(vl_sift_pix) * nel * 2
                        * (f->s_max - f->s_min    )  ) ;
#endif

  f-> sigman  = 0.5 ;
  f-> sigmak  = pow (2.0, 1.0 / nlevels) ;
  f-> sigma0  = 1.6 * f->sigmak ;
  f-> dsigma0 = f->sigma0 * sqrt (1.0 - 1.0 / (f->sigmak*f->sigmak)) ;

  f-> gaussFilter = NULL ;
  f-> gaussFilterSize = 0;
  f-> gaussFilterSigma = 0 ;
  f-> gaussFilterWidth = 0 ;

  f-> octave_width  = 0 ;
  f-> octave_height = 0 ;

  f-> keys     = 0 ;
  f-> nkeys    = 0 ;
  f-> keys_res = 0 ;

  f-> peak_thresh = 0.0 ;
  f-> edge_thresh = 10.0 ;
  f-> norm_thresh = 0.0 ;
  f-> magnif      = 3.0 ;
  f-> windowSize  = NBP / 2 ;

  f-> grad_o  = o_min - 1 ;

  /* initialize fast_expn stuff */
  fast_expn_init (f) ;

  return f ;
}

/** -------------------------------------------------------------------
 ** @brief Delete SIFT filter
 **
 ** @param f SIFT filter to delete.
 **
 ** The function frees the resources allocated by ::vl_sift_new().
 **/

VL_EXPORT
void
vl_sift_delete (VlSiftFilt* f)
{
  if (f) {
    if (f->keys) vl_free (f->keys) ;
#ifndef REDUCE_MEMORY
    if (f->grad) vl_free (f->grad) ;
#endif
    if (f->dog) vl_free (f->dog) ;
    if (f->octave) vl_free (f->octave) ;
#ifndef REDUCE_MEMORY
    if (f->temp) vl_free (f->temp) ;
#endif
    if (f->gaussFilter) vl_free (f->gaussFilter) ;
    vl_free (f) ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Start processing a new image
 **
 ** @param f  SIFT filter.
 ** @param im image data.
 **
 ** The function starts processing a new image by computing its
 ** Gaussian scale space at the lower octave. It also empties the
 ** internal keypoint buffer.
 **
 ** @return error code. The function returns ::VL_ERR_EOF if there are
 ** no more octaves to process.
 **
 ** @sa ::vl_sift_process_next_octave().
 **/

VL_EXPORT
int
vl_sift_process_first_octave (VlSiftFilt *f, vl_sift_pix const *im)
{
  int o, s, h, w ;
  double sa, sb ;
  vl_sift_pix *octave ;

  /* shortcuts */
  vl_sift_pix *temp   = f-> temp ;
  int width           = f-> width ;
  int height          = f-> height ;
  int o_min           = f-> o_min ;
  int s_min           = f-> s_min ;
  int s_max           = f-> s_max ;
  double sigma0       = f-> sigma0 ;
  double sigmak       = f-> sigmak ;
  double sigman       = f-> sigman ;
  double dsigma0      = f-> dsigma0 ;

  /* restart from the first */
  f->o_cur = o_min ;
  f->nkeys = 0 ;
  w = f-> octave_width  = VL_SHIFT_LEFT(f->width,  - f->o_cur) ;
  h = f-> octave_height = VL_SHIFT_LEFT(f->height, - f->o_cur) ;

  /* is there at least one octave? */
  if (f->O == 0)
    return VL_ERR_EOF ;

  /* ------------------------------------------------------------------
   *                     Compute the first sublevel of the first octave
   * --------------------------------------------------------------- */

  /*
   * If the first octave has negative index, we upscale the image; if
   * the first octave has positive index, we downscale the image; if
   * the first octave has index zero, we just copy the image.
   */

  octave = vl_sift_get_octave (f, s_min) ;

  if (o_min < 0) {
    /* double once */
    copy_and_upsample_rows (temp,   im,   width,      height) ;
    copy_and_upsample_rows (octave, temp, height, 2 * width ) ;

    /* double more */
    for(o = -1 ; o > o_min ; --o) {
      copy_and_upsample_rows (temp, octave,
                              width << -o,      height << -o ) ;
      copy_and_upsample_rows (octave, temp,
                              width << -o, 2 * (height << -o)) ;
    }
  }
  else if (o_min > 0) {
    /* downsample */
    copy_and_downsample (octave, im, width, height, o_min) ;
  }
  else {
    /* direct copy */
    memcpy(octave, im, sizeof(vl_sift_pix) * width * height) ;
  }

  /*
   * Here we adjust the smoothing of the first level of the octave.
   * The input image is assumed to have nominal smoothing equal to
   * f->simgan.
   */

  sa = sigma0 * pow (sigmak,   s_min) ;
  sb = sigman * pow (2.0,    - o_min) ;

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb) ;
    _vl_sift_smooth (f, octave, temp, octave, w, h, sd) ;
  }

  /* -----------------------------------------------------------------
   *                                          Compute the first octave
   * -------------------------------------------------------------- */

  for(s = s_min + 1 ; s <= s_max ; ++s) {
    double sd = dsigma0 * pow (sigmak, s) ;
    _vl_sift_smooth (f, vl_sift_get_octave(f, s), temp,
                     vl_sift_get_octave(f, s - 1), w, h, sd) ;
  }

  return VL_ERR_OK ;
}

/** ------------------------------------------------------------------
 ** @brief Process next octave
 **
 ** @param f SIFT filter.
 **
 ** The function computes the next octave of the Gaussian scale space.
 ** Notice that this clears the record of any feature detected in the
 ** previous octave.
 **
 ** @return error code. The function returns the error
 ** ::VL_ERR_EOF when there are no more octaves to process.
 **
 ** @sa ::vl_sift_process_first_octave().
 **/

VL_EXPORT
int
vl_sift_process_next_octave (VlSiftFilt *f)
{

  int s, h, w, s_best ;
  double sa, sb ;
  vl_sift_pix *octave, *pt ;

  /* shortcuts */
  vl_sift_pix *temp   = f-> temp ;
  int O               = f-> O ;
  int S               = f-> S ;
  int o_min           = f-> o_min ;
  int s_min           = f-> s_min ;
  int s_max           = f-> s_max ;
  double sigma0       = f-> sigma0 ;
  double sigmak       = f-> sigmak ;
  double dsigma0      = f-> dsigma0 ;

  /* is there another octave ? */
  if (f->o_cur == o_min + O - 1)
    return VL_ERR_EOF ;

  /* retrieve base */
  s_best = VL_MIN(s_min + S, s_max) ;
  w      = vl_sift_get_octave_width  (f) ;
  h      = vl_sift_get_octave_height (f) ;
  pt     = vl_sift_get_octave        (f, s_best) ;
  octave = vl_sift_get_octave        (f, s_min) ;

  /* next octave */
  copy_and_downsample (octave, pt, w, h, 1) ;

  f-> o_cur            += 1 ;
  f-> nkeys             = 0 ;
  w = f-> octave_width  = VL_SHIFT_LEFT(f->width,  - f->o_cur) ;
  h = f-> octave_height = VL_SHIFT_LEFT(f->height, - f->o_cur) ;

  sa = sigma0 * powf (sigmak, s_min     ) ;
  sb = sigma0 * powf (sigmak, s_best - S) ;

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb) ;
    _vl_sift_smooth (f, octave, temp, octave, w, h, sd) ;
  }

  /* ------------------------------------------------------------------
   *                                                        Fill octave
   * --------------------------------------------------------------- */

  for(s = s_min + 1 ; s <= s_max ; ++s) {
    double sd = dsigma0 * pow (sigmak, s) ;
    _vl_sift_smooth (f, vl_sift_get_octave(f, s), temp,
                     vl_sift_get_octave(f, s - 1), w, h, sd) ;
  }

  return VL_ERR_OK ;
}

static inline void DogSubtractSse2(
  vl_sift_pix* dst,
  const vl_sift_pix* a,
  const vl_sift_pix* b,
  int count)
{
  int i = 0;
  int simdEnd = count & ~3;

  for (; i < simdEnd; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 vd = _mm_sub_ps(vb, va);
    _mm_storeu_ps(dst + i, vd);
  }

  for (; i < count; ++i) {
    dst[i] = b[i] - a[i];
  }
}

void DogSubtract130TileAvx2(
  float* VL_RESTRICT dst,
  const float* VL_RESTRICT a,
  const float* VL_RESTRICT b,
  int height,
  int stride);

static __forceinline void DogSubtract(
  vl_sift_pix* VL_RESTRICT dst,
  const vl_sift_pix* VL_RESTRICT a,
  const vl_sift_pix* VL_RESTRICT b,
  int width,
  int height,
  int stride)
{
  if (hasAVX2 && width == 130) {
    // AVX2 owns the vertical loop
    DogSubtract130TileAvx2(
      dst,
      a,
      b,
      height,
      stride);
    return;
  }

  // Fallbacks: row-based
  for (int y = 0; y < height; ++y) {
    if (hasAVX2) {
      DogSubtractAvx2(dst, a, b, width);
    }
    else {
      DogSubtractSse2(dst, a, b, width);
    }

    dst += stride;
    a += stride;
    b += stride;
  }
}

/** ------------------------------------------------------------------
 ** @brief Detect keypoints
 **
 ** The function detect keypoints in the current octave filling the
 ** internal keypoint buffer. Keypoints can be retrieved by
 ** ::vl_sift_get_keypoints().
 **
 ** @param f SIFT filter.
 **/

#undef DO_ORIGINAL
#undef VALIDATE
#undef VALIDATE2

#ifdef VALIDATE2
#pragma optimize("", off) // JPB WIP BUG
#endif
VL_EXPORT
void
vl_sift_detect (VlSiftFilt * f)
{
  VlSiftFilt* start = f;

#ifdef DO_ORIGINAL
  for (int pass = 1; pass < 2; ++pass) {
#else
#ifdef VALIDATE
  int pass0Keys = 0;
  for (int pass = 0; pass < 2; ++pass) {
#else
  for (int pass = 0; pass < 1; ++pass) {
#endif
#endif
    if (pass == 0) { // OPT
      vl_sift_pix* dog = f->dog;
      int          s_min = f->s_min;
      int          s_max = f->s_max;
      int          w = f->octave_width;
      int          h = f->octave_height;
      double       te = f->edge_thresh;
      double       tp = f->peak_thresh;

      int const    xo = 1;      /* x-stride */
      int const    yo = w;      /* y-stride */
      int const    so = w * h;  /* s-stride */

      double       xper = pow(2.0, f->o_cur);

      int x, y, s, i, ii, jj;
      vl_sift_pix const* __restrict pt;
      vl_sift_pix v;
      VlSiftKeypoint* __restrict k;

      /* clear current list */
      f->nkeys = 0;

      /* -----------------------------------------------------------------
       *                                          Find local maxima of DoG
       * -------------------------------------------------------------- */

      float const tolerance = 0.8 * tp;

      int const tileHeight = 16;
      int const tileWidth = 128;//192*4; /* Determined empirically */
      int const imageInteriorHeight = ((h - 1) - 1);
      int const imageInteriorWidth = ((w - 1) - 1);
      int numTilesY = ceil(((double)imageInteriorHeight) / tileHeight);
      int numTilesX = ceil(((double)imageInteriorWidth) / tileWidth);

      /* ------------------------------------------------------------
     * Build full DoG stack (VLFeat semantics)
     * ------------------------------------------------------------ */
      vl_sift_pix* dogPtr = dog;
      for (int ss = s_min; ss <= s_max - 1; ++ss) {
        const vl_sift_pix* a = vl_sift_get_octave(f, ss);
        const vl_sift_pix* b = vl_sift_get_octave(f, ss + 1);

        // Full image DoG for this scale
        DogSubtract(
          dogPtr,   // dst
          a,        // octave[s]
          b,        // octave[s+1]
          w,        // width
          h,        // height
          w         // stride
        );

        dogPtr += so; // advance one DoG slice
      }

      /* Detect keypoints in blocks.  Block size is designed to maximize cache reuse. */
      for (s = s_min + 1; s <= s_max - 2; ++s) {
        vl_sift_pix* __restrict prevDog = dog + so * ((s - 1) - s_min);
        vl_sift_pix* __restrict curDog = dog + so * (s - s_min);
        vl_sift_pix* __restrict nextDog = dog + so * ((s + 1) - s_min);

        for (int ty = 0; ty < imageInteriorHeight; ty += tileHeight) {

          int actualTileHeight = min(tileHeight, imageInteriorHeight - ty);

          for (int tx = 0; tx < imageInteriorWidth; tx += tileWidth) {

            int actualTileWidth = min(tileWidth, imageInteriorWidth - tx);
            int tileIndexBase = (ty + 1) * w + (tx + 1);

            const vl_sift_pix* __restrict ptRow = curDog + tileIndexBase;

            for (int y = 0; y < actualTileHeight; ++y) {

              const vl_sift_pix* __restrict pt = ptRow;
              int globalY = ty + y + 1;

              for (int x = 0; x < actualTileWidth; ++x, ++pt) {
                float v = *pt;

                ptrdiff_t idx = pt - curDog;

                const vl_sift_pix* nextBase = nextDog + idx;
                const vl_sift_pix* prevBase = prevDog + idx;

                if (
                  (v >= tolerance &&
                    v >= pt[xo] &&
                    v >= pt[-xo] &&
                    v >= nextBase[0] &&
                    v >= prevBase[0] &&
                    v >= pt[yo] &&
                    v >= pt[-yo] &&
                    v >= pt[yo + xo] &&
                    v >= pt[yo - xo] &&
                    v >= pt[-yo + xo] &&
                    v >= pt[-yo - xo] &&
                    v >= nextBase[xo] &&
                    v >= nextBase[-xo] &&
                    v >= nextBase[yo] &&
                    v >= nextBase[-yo] &&
                    v >= nextBase[yo + xo] &&
                    v >= nextBase[yo - xo] &&
                    v >= nextBase[-yo + xo] &&
                    v >= nextBase[-yo - xo] &&
                    v >= prevBase[xo] &&
                    v >= prevBase[-xo] &&
                    v >= prevBase[yo] &&
                    v >= prevBase[-yo] &&
                    v >= prevBase[yo + xo] &&
                    v >= prevBase[yo - xo] &&
                    v >= prevBase[-yo + xo] &&
                    v >= prevBase[-yo - xo]
                    ) ||
                  (v <= -tolerance &&
                    v <= pt[xo] &&
                    v <= pt[-xo] &&
                    v <= nextBase[0] &&
                    v <= prevBase[0] &&
                    v <= pt[yo] &&
                    v <= pt[-yo] &&
                    v <= pt[yo + xo] &&
                    v <= pt[yo - xo] &&
                    v <= pt[-yo + xo] &&
                    v <= pt[-yo - xo] &&
                    v <= nextBase[xo] &&
                    v <= nextBase[-xo] &&
                    v <= nextBase[yo] &&
                    v <= nextBase[-yo] &&
                    v <= nextBase[yo + xo] &&
                    v <= nextBase[yo - xo] &&
                    v <= nextBase[-yo + xo] &&
                    v <= nextBase[-yo - xo] &&
                    v <= prevBase[xo] &&
                    v <= prevBase[-xo] &&
                    v <= prevBase[yo] &&
                    v <= prevBase[-yo] &&
                    v <= prevBase[yo + xo] &&
                    v <= prevBase[yo - xo] &&
                    v <= prevBase[-yo + xo] &&
                    v <= prevBase[-yo - xo]
                    )
                  ) {
                  goto is_keypoint;
                }
                continue;
#ifdef VALIDATE2
                static int printCount = 0;
                if (printCount < 10) {
                  printf("OPT  s=%d y=%d x=%d\n", s, globalY, globalX);
                  ++printCount;
                }
#endif

              is_keypoint:
                /* make room for more keypoints */
                if (f->nkeys >= f->keys_res) {
                  f->keys_res += 32768;
                  if (f->keys) {
                    f->keys = vl_realloc(
                      f->keys,
                      f->keys_res * sizeof(VlSiftKeypoint)
                    );
                  }
                  else {
                    f->keys = vl_malloc(
                      f->keys_res * sizeof(VlSiftKeypoint)
                    );
                  }
                }

                k = f->keys + (f->nkeys++);
                k->ix = tx + x + 1;
                k->iy = ty + y + 1;
                k->is = s;
              }

              ptRow += w;
            }
          } // tx
        } // ty
      } // s

      /* -----------------------------------------------------------------
       *                                               Refine local maxima
       * -------------------------------------------------------------- */

       /* this pointer is used to write the keypoints back */
      k = f->keys;

      for (i = 0; i < f->nkeys; ++i) {
        int x = f->keys[i].ix;
        int y = f->keys[i].iy;
        int s = f->keys[i].is;

        double Dx = 0, Dy = 0, Ds = 0, Dxx = 0, Dyy = 0, Dss = 0, Dxy = 0, Dxs = 0, Dys = 0;
        double A[3 * 3], b[3];

        int dx = 0;
        int dy = 0;

        int iter, i, j;

        for (iter = 0; iter < 5; ++iter) {
          x += dx;
          y += dy;

          pt = dog
            + xo * x
            + yo * y
            + so * (s - s_min);

          /** @brief Index GSS @internal */
#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))

      /** @brief Index matrix A @internal */
#define Aat(i,j)     (A[(i)+(j)*3])

      /* compute the gradient */
          Dx = 0.5 * (at(+1, 0, 0) - at(-1, 0, 0));
          Dy = 0.5 * (at(0, +1, 0) - at(0, -1, 0));
          Ds = 0.5 * (at(0, 0, +1) - at(0, 0, -1));

          /* compute the Hessian */
          Dxx = (at(+1, 0, 0) + at(-1, 0, 0) - 2.0 * at(0, 0, 0));
          Dyy = (at(0, +1, 0) + at(0, -1, 0) - 2.0 * at(0, 0, 0));
          Dss = (at(0, 0, +1) + at(0, 0, -1) - 2.0 * at(0, 0, 0));

          Dxy = 0.25 * (at(+1, +1, 0) + at(-1, -1, 0) - at(-1, +1, 0) - at(+1, -1, 0));
          Dxs = 0.25 * (at(+1, 0, +1) + at(-1, 0, -1) - at(-1, 0, +1) - at(+1, 0, -1));
          Dys = 0.25 * (at(0, +1, +1) + at(0, -1, -1) - at(0, -1, +1) - at(0, +1, -1));

          /* solve linear system ....................................... */
          Aat(0, 0) = Dxx;
          Aat(1, 1) = Dyy;
          Aat(2, 2) = Dss;
          Aat(0, 1) = Aat(1, 0) = Dxy;
          Aat(0, 2) = Aat(2, 0) = Dxs;
          Aat(1, 2) = Aat(2, 1) = Dys;

          b[0] = -Dx;
          b[1] = -Dy;
          b[2] = -Ds;

          /* Gauss elimination */
          for (j = 0; j < 3; ++j) {
            double maxa = 0;
            double maxabsa = 0;
            int    maxi = -1;
            double tmp;

            /* look for the maximally stable pivot */
            for (i = j; i < 3; ++i) {
              double a = Aat(i, j);
              double absa = vl_abs_d(a);
              if (absa > maxabsa) {
                maxa = a;
                maxabsa = absa;
                maxi = i;
              }
            }

            /* if singular give up */
            if (maxabsa < 1e-10f) {
              b[0] = 0;
              b[1] = 0;
              b[2] = 0;
              break;
            }

            i = maxi;

            /* swap j-th row with i-th row and normalize j-th row */
            for (jj = j; jj < 3; ++jj) {
              tmp = Aat(i, jj); Aat(i, jj) = Aat(j, jj); Aat(j, jj) = tmp;
              Aat(j, jj) /= maxa;
            }
            tmp = b[j]; b[j] = b[i]; b[i] = tmp;
            b[j] /= maxa;

            /* elimination */
            for (ii = j + 1; ii < 3; ++ii) {
              double x = Aat(ii, j);
              for (jj = j; jj < 3; ++jj) {
                Aat(ii, jj) -= x * Aat(j, jj);
              }
              b[ii] -= x * b[j];
            }
          }

          /* backward substitution */
          for (i = 2; i > 0; --i) {
            double x = b[i];
            for (ii = i - 1; ii >= 0; --ii) {
              b[ii] -= x * Aat(ii, i);
            }
          }

          /* .......................................................... */
          /* If the translation of the keypoint is big, move the keypoint
           * and re-iterate the computation. Otherwise we are all set.
           */

          dx = ((b[0] > 0.6 && x < w - 2) ? 1 : 0)
            + ((b[0] < -0.6 && x > 1) ? -1 : 0);

          dy = ((b[1] > 0.6 && y < h - 2) ? 1 : 0)
            + ((b[1] < -0.6 && y > 1) ? -1 : 0);

          if (dx == 0 && dy == 0) break;
        }

        /* check threshold and other conditions */
        {
          double val = at(0, 0, 0)
            + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]);
          double score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
          double xn = x + b[0];
          double yn = y + b[1];
          double sn = s + b[2];

          vl_bool good =
            vl_abs_d(val) > tp &&
            score < (te + 1) * (te + 1) / te &&
            score >= 0 &&
            vl_abs_d(b[0]) < 1.5 &&
            vl_abs_d(b[1]) < 1.5 &&
            vl_abs_d(b[2]) < 1.5 &&
            xn >= 0 &&
            xn <= w - 1 &&
            yn >= 0 &&
            yn <= h - 1 &&
            sn >= s_min &&
            sn <= s_max;

          if (good) {
            k->o = f->o_cur;
            k->ix = x;
            k->iy = y;
            k->is = s;
            k->s = sn;
            k->x = xn * xper;
            k->y = yn * xper;
            k->sigma = f->sigma0 * pow(2.0, sn / f->S) * xper;
            ++k;
          }
        }
      } /* next keypoint to refine */

      /* update keypoint count */
      f->nkeys = (int)(k - f->keys);
#ifdef VALIDATE
      pass0Keys = f->nkeys;
#endif
#ifdef VALIDATE2
      printf("OPT final keys: %d\n", f->nkeys);
#endif
    }
    else { //REF
      f = start; // restart
      vl_sift_pix* dog = f->dog;
      int          s_min = f->s_min;
      int          s_max = f->s_max;
      int          w = f->octave_width;
      int          h = f->octave_height;
      double       te = f->edge_thresh;
      double       tp = f->peak_thresh;

      int const    xo = 1;      /* x-stride */
      int const    yo = w;      /* y-stride */
      int const    so = w * h;  /* s-stride */

      double       xper = pow(2.0, f->o_cur);

      int x, y, s, i, ii, jj;
      vl_sift_pix* pt, v;
      VlSiftKeypoint* k;

      /* clear current list */
      f->nkeys = 0;

      /* compute difference of gaussian (DoG) */
      pt = f->dog;
      for (s = s_min; s <= s_max - 1; ++s) {
        vl_sift_pix* src_a = vl_sift_get_octave(f, s);
        vl_sift_pix* src_b = vl_sift_get_octave(f, s + 1);
        vl_sift_pix* end_a = src_a + w * h;
        while (src_a != end_a) {
#if 0 // JPB WIP BUG are identical
          float new = *src_b - *src_a;
          if (new != *pt) {
            __debugbreak();
          }
#endif
          *pt++ = *src_b++ - *src_a++;
        }
      }

      /* -----------------------------------------------------------------
       *                                          Find local maxima of DoG
       * -------------------------------------------------------------- */

       /* start from dog [1,1,s_min+1] */
      pt = dog + xo + yo + so;

      for (s = s_min + 1; s <= s_max - 2; ++s) {
        for (y = 1; y < h - 1; ++y) {
          for (x = 1; x < w - 1; ++x) {
            v = *pt;

#define CHECK_NEIGHBORS(CMP,SGN)                    \
        ( v CMP ## = SGN 0.8 * tp &&                \
          v CMP *(pt + xo) &&                       \
          v CMP *(pt - xo) &&                       \
          v CMP *(pt + so) &&                       \
          v CMP *(pt - so) &&                       \
          v CMP *(pt + yo) &&                       \
          v CMP *(pt - yo) &&                       \
                                                    \
          v CMP *(pt + yo + xo) &&                  \
          v CMP *(pt + yo - xo) &&                  \
          v CMP *(pt - yo + xo) &&                  \
          v CMP *(pt - yo - xo) &&                  \
                                                    \
          v CMP *(pt + xo      + so) &&             \
          v CMP *(pt - xo      + so) &&             \
          v CMP *(pt + yo      + so) &&             \
          v CMP *(pt - yo      + so) &&             \
          v CMP *(pt + yo + xo + so) &&             \
          v CMP *(pt + yo - xo + so) &&             \
          v CMP *(pt - yo + xo + so) &&             \
          v CMP *(pt - yo - xo + so) &&             \
                                                    \
          v CMP *(pt + xo      - so) &&             \
          v CMP *(pt - xo      - so) &&             \
          v CMP *(pt + yo      - so) &&             \
          v CMP *(pt - yo      - so) &&             \
          v CMP *(pt + yo + xo - so) &&             \
          v CMP *(pt + yo - xo - so) &&             \
          v CMP *(pt - yo + xo - so) &&             \
          v CMP *(pt - yo - xo - so) )

            if (CHECK_NEIGHBORS(> , +) ||
              CHECK_NEIGHBORS(< , -)) {
#ifdef VALIDAE2
              static int printCountRef = 0;
              if (printCountRef < 10) {
                printf("REF  s=%d y=%d x=%d\n", s, y, x);
                ++printCountRef;
              }
#endif

              /* make room for more keypoints */
              if (f->nkeys >= f->keys_res) {
                f->keys_res += 500;
                if (f->keys) {
                  f->keys = vl_realloc(f->keys,
                    f->keys_res *
                    sizeof(VlSiftKeypoint));
                }
                else {
                  f->keys = vl_malloc(f->keys_res *
                    sizeof(VlSiftKeypoint));
                }
              }

              k = f->keys + (f->nkeys++);

              k->ix = x;
              k->iy = y;
              k->is = s;
            }
            pt += 1;
          }
          pt += 2;
        }
        pt += 2 * yo;
      }

      /* -----------------------------------------------------------------
       *                                               Refine local maxima
       * -------------------------------------------------------------- */

       /* this pointer is used to write the keypoints back */
      k = f->keys;

      for (i = 0; i < f->nkeys; ++i) {
        int x = f->keys[i].ix;
        int y = f->keys[i].iy;
        int s = f->keys[i].is;

        double Dx = 0, Dy = 0, Ds = 0, Dxx = 0, Dyy = 0, Dss = 0, Dxy = 0, Dxs = 0, Dys = 0;
        double A[3 * 3], b[3];

        int dx = 0;
        int dy = 0;

        int iter, i, j;

        for (iter = 0; iter < 5; ++iter) {

          x += dx;
          y += dy;

          pt = dog
            + xo * x
            + yo * y
            + so * (s - s_min);

          /** @brief Index GSS @internal */
#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))

      /** @brief Index matrix A @internal */
#define Aat(i,j)     (A[(i)+(j)*3])

      /* compute the gradient */
          Dx = 0.5 * (at(+1, 0, 0) - at(-1, 0, 0));
          Dy = 0.5 * (at(0, +1, 0) - at(0, -1, 0));
          Ds = 0.5 * (at(0, 0, +1) - at(0, 0, -1));

          /* compute the Hessian */
          Dxx = (at(+1, 0, 0) + at(-1, 0, 0) - 2.0 * at(0, 0, 0));
          Dyy = (at(0, +1, 0) + at(0, -1, 0) - 2.0 * at(0, 0, 0));
          Dss = (at(0, 0, +1) + at(0, 0, -1) - 2.0 * at(0, 0, 0));

          Dxy = 0.25 * (at(+1, +1, 0) + at(-1, -1, 0) - at(-1, +1, 0) - at(+1, -1, 0));
          Dxs = 0.25 * (at(+1, 0, +1) + at(-1, 0, -1) - at(-1, 0, +1) - at(+1, 0, -1));
          Dys = 0.25 * (at(0, +1, +1) + at(0, -1, -1) - at(0, -1, +1) - at(0, +1, -1));

          /* solve linear system ....................................... */
          Aat(0, 0) = Dxx;
          Aat(1, 1) = Dyy;
          Aat(2, 2) = Dss;
          Aat(0, 1) = Aat(1, 0) = Dxy;
          Aat(0, 2) = Aat(2, 0) = Dxs;
          Aat(1, 2) = Aat(2, 1) = Dys;

          b[0] = -Dx;
          b[1] = -Dy;
          b[2] = -Ds;

          /* Gauss elimination */
          for (j = 0; j < 3; ++j) {
            double maxa = 0;
            double maxabsa = 0;
            int    maxi = -1;
            double tmp;

            /* look for the maximally stable pivot */
            for (i = j; i < 3; ++i) {
              double a = Aat(i, j);
              double absa = vl_abs_d(a);
              if (absa > maxabsa) {
                maxa = a;
                maxabsa = absa;
                maxi = i;
              }
            }

            /* if singular give up */
            if (maxabsa < 1e-10f) {
              b[0] = 0;
              b[1] = 0;
              b[2] = 0;
              break;
            }

            i = maxi;

            /* swap j-th row with i-th row and normalize j-th row */
            for (jj = j; jj < 3; ++jj) {
              tmp = Aat(i, jj); Aat(i, jj) = Aat(j, jj); Aat(j, jj) = tmp;
              Aat(j, jj) /= maxa;
            }
            tmp = b[j]; b[j] = b[i]; b[i] = tmp;
            b[j] /= maxa;

            /* elimination */
            for (ii = j + 1; ii < 3; ++ii) {
              double x = Aat(ii, j);
              for (jj = j; jj < 3; ++jj) {
                Aat(ii, jj) -= x * Aat(j, jj);
              }
              b[ii] -= x * b[j];
            }
          }

          /* backward substitution */
          for (i = 2; i > 0; --i) {
            double x = b[i];
            for (ii = i - 1; ii >= 0; --ii) {
              b[ii] -= x * Aat(ii, i);
            }
          }

          /* .......................................................... */
          /* If the translation of the keypoint is big, move the keypoint
           * and re-iterate the computation. Otherwise we are all set.
           */

          dx = ((b[0] > 0.6 && x < w - 2) ? 1 : 0)
            + ((b[0] < -0.6 && x > 1) ? -1 : 0);

          dy = ((b[1] > 0.6 && y < h - 2) ? 1 : 0)
            + ((b[1] < -0.6 && y > 1) ? -1 : 0);

          if (dx == 0 && dy == 0) break;
        }

        /* check threshold and other conditions */
        {
          double val = at(0, 0, 0)
            + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]);
          double score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
          double xn = x + b[0];
          double yn = y + b[1];
          double sn = s + b[2];

          vl_bool good =
            vl_abs_d(val) > tp &&
            score < (te + 1) * (te + 1) / te &&
            score >= 0 &&
            vl_abs_d(b[0]) < 1.5 &&
            vl_abs_d(b[1]) < 1.5 &&
            vl_abs_d(b[2]) < 1.5 &&
            xn >= 0 &&
            xn <= w - 1 &&
            yn >= 0 &&
            yn <= h - 1 &&
            sn >= s_min &&
            sn <= s_max;

          if (good) {
            k->o = f->o_cur;
            k->ix = x;
            k->iy = y;
            k->is = s;
            k->s = sn;
            k->x = xn * xper;
            k->y = yn * xper;
            k->sigma = f->sigma0 * pow(2.0, sn / f->S) * xper;
            ++k;
          }

        } /* done checking */
      } /* next keypoint to refine */

      f->nkeys = (int)(k - f->keys);
#ifdef VALIDATE2
      printf("REF final keys: %d\n", f->nkeys);
#endif
    }

#ifdef VALIDATE2
    if (pass == 0) {
      FILE* ff = fopen("opt_candidates.txt", "w");
      for (int i = 0; i < f->nkeys; ++i)
        fprintf(ff, "%d %d %d\n", f->keys[i].is, f->keys[i].iy, f->keys[i].ix);
      fclose(ff);
    }
    else {
      FILE* ff = fopen("ref_candidates.txt", "w");
      for (int i = 0; i < f->nkeys; ++i)
        fprintf(ff, "%d %d %d\n", f->keys[i].is, f->keys[i].iy, f->keys[i].ix);
      fclose(ff);
    }
#endif
  }
#ifdef VALIDATE
  if (pass0Keys != f->nkeys) {
    printf("Validation failed: keypoint count mismatch. OPT %d vs REF %d\n", pass0Keys, f->nkeys);
  }
#endif
}

/** ------------------------------------------------------------------
 ** @brief Update gradients to current GSS octave
 **
 ** @param f SIFT filter.
 **
 ** The function makes sure that the gradient buffer is up-to-date
 ** with the current GSS data.
 **
 ** @remark The minimum octave size is 2x2xS.
 **/

#if FAST_SIFT_GRADIENT_UPDATE
#if 1
static inline void FastGradientRow(
  vl_sift_pix const** g,
  vl_sift_pix const** __restrict s,
  vl_sift_pix const* const __restrict end,
  int xo,
  int yo
)
{
  vl_sift_pix* __restrict src = (vl_sift_pix*)*s;
  vl_sift_pix* __restrict grad = (vl_sift_pix*)*g;

  const int count = (int)(end - src);
  const int numGroups = count / GROUP_SIZE;

  const _Data vHalf = _Set(0.5f);

  int i = 0;

  // Tune these. 2 groups ahead is a decent starting point.
  const int pfGroups = 2;
  const int pfDist = pfGroups * GROUP_SIZE;

  // Unroll by 2 groups
  for (; i + 1 < numGroups; i += 2) {
    vl_sift_pix* __restrict p0 = src + 0 * GROUP_SIZE;
    vl_sift_pix* __restrict p1 = src + 1 * GROUP_SIZE;
    // ---- group 0 ----
    {
      const _Data vPx = _Load(p0 + xo);
      const _Data vMx = _Load(p0 - xo);
      const _Data vPy = _Load(p0 + yo);
      const _Data vMy = _Load(p0 - yo);

      const _Data vHalfGx = _Mul(vHalf, _Sub(vPx, vMx));
      const _Data vHalfGy = _Mul(vHalf, _Sub(vPy, vMy));

      // Norm = gx^2 + gy^2
      const _Data vNorm = _Add(_Mul(vHalfGx, vHalfGx), _Mul(vHalfGy, vHalfGy));

      const _Data vMag = _Sqrt(vNorm);
      const _Data vAng = GradCalc2(vHalfGy, vHalfGx);

      _Store(grad + 0, _UnpackLow(vMag, vAng));
      _Store(grad + GROUP_SIZE, _UnpackHigh(vMag, vAng));
    }

    // ---- group 1 ----
    {
      const _Data vPx = _Load(p1 + xo);
      const _Data vMx = _Load(p1 - xo);
      const _Data vPy = _Load(p1 + yo);
      const _Data vMy = _Load(p1 - yo);

      const _Data vHalfGx = _Mul(vHalf, _Sub(vPx, vMx));
      const _Data vHalfGy = _Mul(vHalf, _Sub(vPy, vMy));

      const _Data vNorm = _Add(_Mul(vHalfGx, vHalfGx), _Mul(vHalfGy, vHalfGy));

      const _Data vMag = _Sqrt(vNorm);
      const _Data vAng = GradCalc2(vHalfGy, vHalfGx);

      vl_sift_pix* __restrict g1 = grad + GROUP_SIZE * 2;
      _Store(g1 + 0, _UnpackLow(vMag, vAng));
      _Store(g1 + GROUP_SIZE, _UnpackHigh(vMag, vAng));
    }

    src += 2 * GROUP_SIZE;
    grad += 2 * GROUP_SIZE * 2;
  }

  // Tail group (if odd)
  for (; i < numGroups; ++i, src += GROUP_SIZE, grad += GROUP_SIZE * 2) {
    const _Data vPx = _Load(src + xo);
    const _Data vMx = _Load(src - xo);
    const _Data vPy = _Load(src + yo);
    const _Data vMy = _Load(src - yo);

    const _Data vHalfGx = _Mul(vHalf, _Sub(vPx, vMx));
    const _Data vHalfGy = _Mul(vHalf, _Sub(vPy, vMy));

    const _Data vNorm = _Add(_Mul(vHalfGx, vHalfGx), _Mul(vHalfGy, vHalfGy));
    const _Data vMag = _Sqrt(vNorm);
    const _Data vAng = GradCalc2(vHalfGy, vHalfGx);

    _Store(grad + 0, _UnpackLow(vMag, vAng));
    _Store(grad + GROUP_SIZE, _UnpackHigh(vMag, vAng));
  }

  *s = (vl_sift_pix const*)src;
  *g = (vl_sift_pix const*)grad;
}
#else
void FastGradientRow(
  vl_sift_pix const ** g,
  vl_sift_pix const ** __restrict s,
  vl_sift_pix const * const __restrict end,
  int xo,
  int yo
)
{
  vl_sift_pix* __restrict src = *s;
  vl_sift_pix* __restrict grad = *g;

  /* Perform the processing GROUP_SIZE elements at a time. */
  int const count       = end - src;
  int const numGroups   = count / GROUP_SIZE;
  _Data const vHalf     = _Set(0.5f);

  for (int i = 0; i < numGroups; ++i, src += GROUP_SIZE, grad += GROUP_SIZE * 2) {
    _Data const vGx     = _Load(src + xo); // s
    _Data const vGy     = _Load(src + yo);
    _Data const vGx2    = _Load(src - xo);
    _Data const vGy2    = _Load(src - yo);
                         
    _Data const vDiff   = _Sub(vGx, vGx2);
    _Data const vDiff2  = _Sub(vGy, vGy2);

    /* Calculate the first half of the gradient. */
    _Data const vHalfGx = _Mul(vHalf, vDiff);
    _Data const vHalfGy = _Mul(vHalf, vDiff2);
    _Data const vNorm   = _Add(_Mul(vHalfGx, vHalfGx), _Mul(vHalfGy, vHalfGy));
    _Data const vSqrt   = _Sqrt(vNorm);

    /* Now the second. */
    _Data const vAtan2 = GradCalc2(vHalfGy, vHalfGx);

    /* Store s0, s1, s2, s3 and a0, a1, a2, a3 as: */
    _Store( /* s0, a0, s1, a1 */
      grad,
      _UnpackLow(vSqrt, vAtan2)
    );
    _Store( /* s2, a2, s3, a3 */
      grad + GROUP_SIZE,
      _UnpackHigh(vSqrt, vAtan2)
    );
  }

  *s = src;
  *g = grad;
}
#endif
#endif /* FAST_SIFT_GRADIENT_UPDATE */

void
vl_sift_update_gradient (VlSiftFilt *f)
{
  int       s_min = f->s_min ;
  int       s_max = f->s_max ;
  int       w     = vl_sift_get_octave_width  (f) ;
  int       h     = vl_sift_get_octave_height (f) ;
  int const xo    = 1 ;
  int const yo    = w ;
  int const so    = h * w ;
  int y, s ;

  if (f->grad_o == f->o_cur) return ;

  for (s  = s_min + 1 ;
       s <= s_max - 2 ; ++ s) {

    vl_sift_pix *src, *end, *grad, gx, gy ;

#define SAVE_BACK                                                       \
    *grad++ = vl_fast_sqrt_f (gx*gx + gy*gy) ;                          \
    *grad++ = vl_mod_2pi_f   (vl_fast_atan2_f (gy, gx) + 2*VL_PI) ;     \
    ++src ;                                                             \

    src  = vl_sift_get_octave (f,s) ;
    grad = f->grad + 2 * so * (s - s_min -1) ;

    /* first pixel of the first row */
    gx = src[+xo] - src[0] ;
    gy = src[+yo] - src[0] ;
    SAVE_BACK ;

    /* middle pixels of the  first row */
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy =        src[+yo] - src[0] ;
      SAVE_BACK ;
    }

    /* last pixel of the first row */
    gx = src[0]   - src[-xo] ;
    gy = src[+yo] - src[0] ;
    SAVE_BACK ;

    for (y = 1 ; y < h -1 ; ++y) {

      /* first pixel of the middle rows */
      gx =        src[+xo] - src[0] ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;

      /* middle pixels of the middle rows */
      end = (src - 1) + w - 1 ;
#if FAST_SIFT_GRADIENT_UPDATE
      /* Perform the processing on the remaining elements. */
      FastGradientRow( &grad, &src, end, xo, yo );
#endif

      /* middle pixels of the middle rows */
      while (src < end) {
        gx = 0.5 * (src[+xo] - src[-xo]) ;
        gy = 0.5 * (src[+yo] - src[-yo]) ;
        SAVE_BACK ;
      }

      /* last pixel of the middle row */
      gx =        src[0]   - src[-xo] ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;
    }

    /* first pixel of the last row */
    gx = src[+xo] - src[0] ;
    gy = src[  0] - src[-yo] ;
    SAVE_BACK ;

    /* middle pixels of the last row */
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy =        src[0]   - src[-yo] ;
      SAVE_BACK ;
    }

    /* last pixel of the last row */
    gx = src[0]   - src[-xo] ;
    gy = src[0]   - src[-yo] ;
    SAVE_BACK ;
  }
  f->grad_o = f->o_cur ;
}

/** ------------------------------------------------------------------
 ** @brief Calculate the keypoint orientation(s)
 **
 ** @param f        SIFT filter.
 ** @param angles   orientations (output).
 ** @param k        keypoint.
 **
 ** The function computes the orientation(s) of the keypoint @a k.
 ** The function returns the number of orientations found (up to
 ** four). The orientations themselves are written to the vector @a
 ** angles.
 **
 ** @remark The function requires the keypoint octave @a k->o to be
 ** equal to the filter current octave ::vl_sift_get_octave. If this
 ** is not the case, the function returns zero orientations.
 **
 ** @remark The function requires the keypoint scale level @c k->s to
 ** be in the range @c s_min+1 and @c s_max-2 (where usually @c
 ** s_min=0 and @c s_max=S+2). If this is not the case, the function
 ** returns zero orientations.
 **
 ** @return number of orientations found.
 **/

VL_EXPORT
int
vl_sift_calc_keypoint_orientations (VlSiftFilt *f,
                                    double angles [4],
                                    VlSiftKeypoint const *k)
{
  double const winf   = 1.5 ;
  double       xper   = pow (2.0, f->o_cur) ;

  int          w      = f-> octave_width ;
  int          h      = f-> octave_height ;
  int const    xo     = 2 ;         /* x-stride */
  int const    yo     = 2 * w ;     /* y-stride */
  int const    so     = 2 * w * h ; /* s-stride */
  double       x      = k-> x     / xper ;
  double       y      = k-> y     / xper ;
  double       sigma  = k-> sigma / xper ;

  int          xi     = (int) (x + 0.5) ;
  int          yi     = (int) (y + 0.5) ;
  int          si     = k-> is ;

  double const sigmaw = winf * sigma ;
  int          W      = VL_MAX(floor (3.0 * sigmaw), 1) ;

  int          nangles= 0 ;

  enum {nbins = 36} ;

  double hist [nbins], maxh ;
  vl_sift_pix const * pt ;
  int xs, ys, iter, i ;

  /* skip if the keypoint octave is not current */
  if(k->o != f->o_cur)
    return 0 ;

  /* skip the keypoint if it is out of bounds */
  if(xi < 0            ||
     xi > w - 1        ||
     yi < 0            ||
     yi > h - 1        ||
     si < f->s_min + 1 ||
     si > f->s_max - 2  ) {
    return 0 ;
  }

  /* clear histogram */
  memset (hist, 0, sizeof(double) * nbins) ;

  /* compute orientation histogram */
  pt = f-> grad + xo*xi + yo*yi + so*(si - f->s_min - 1) ;

#undef  at
#define at(dx,dy) (*(pt + xo * (dx) + yo * (dy)))

  for(ys  =  VL_MAX (- W,       - yi) ;
      ys <=  VL_MIN (+ W, h - 1 - yi) ; ++ys) {

#if FAST_SIFT_CALC_KEYPOINT_ORIENTATIONS
      JPB WIP
    __declspec(align(ALIGN_SIZE)) int modv[ GROUP_SIZE ];
    __declspec(align(ALIGN_SIZE)) int angv[ GROUP_SIZE ];
    __declspec(align(ALIGN_SIZE)) float r2s[ GROUP_SIZE ];
    int cnt = 0;

    vl_sift_pix mod, ang, fbin;
    const float fbinFactor = ( ( float ) nbins ) / ( 2.f * VL_PI );
    _DataF fbinFactorv = _Set( fbinFactor );
    float invTwoSignmaw = 1.0 / ( 2.0 * sigmaw * sigmaw );
    _DataF invTwoSignmawv = _Set(invTwoSignmaw );
    _DataF half = _Set( 0.5f );
    float tolerance = ( ( ( float ) W * W ) + 0.6f );

    int ysy0 = ys * yo;
    for(xs  = VL_MAX (- W,       - xi) ;
        xs <= VL_MIN (+ W, w - 1 - xi) ; ++xs) {
        float dx = (float)(xi + xs) - x;
        float dy = (float)(yi + ys) - y;
        float r2 = dx*dx + dy*dy ;

        /* limit to a circular window */
        if (r2 >= tolerance ) continue ;

        modv[ cnt ] = *( pt + xs * xo + ysy0 );
        angv[ cnt ] = *( pt + xs * xo + ysy0 + 1 );
        r2s[ cnt ] = r2;

        ++cnt;
        if (GROUP_SIZE == cnt) {
            _DataF wgtv = FastExp(_Mul(_LoadA(r2s), invTwoSignmawv)) ;
            _DataF fbinv = _Mul( _LoadA( angv ), fbinFactorv );

            _DataF fast_floor_rbset( _DataF x );
            _DataF binfv = fast_floor_rbset( _Sub( fbinv, half ) );
            _DataI biniv = _ConvertIF( fbinv );
            _DataF rbin = _Sub( _Sub( fbinv, binfv ), half );
            _DataF modWgt = _Mul( _Load( modv ), wgtv );

            for (int i = 0; i < GROUP_SIZE; ++i) {
                hist[ ( _AsArrayI( biniv, i ) + nbins ) % nbins ] += ( 1 - _AsArrayF( rbin, i ) ) * _AsArrayF( modWgt, i );
                hist[ ( _AsArrayI( biniv, i ) + 1 ) % nbins ] += ( _AsArrayF( rbin, i ) ) * _AsArrayF( modWgt, i );
            }
            cnt = 0;
        }
    }

    for (int i = 0; i < cnt; ++i) {
        mod = *( pt + xs * xo + ysy0 );
        ang= *( pt + xs * xo + ysy0 + 1 );

        fbin = ang * fbinFactor;

        float wgt  = FastExp ( -r2s[ i ] * invTwoSignmaw) ;
        int bin = ( int ) vl_floor_f( fbin - 0.5f );
        float rbin = fbin - bin - 0.5;
        float modWgt = mod * wgt;
        hist[ ( bin + nbins ) % nbins ] += ( 1 - rbin ) * modWgt;
        hist[ ( bin + 1 ) % nbins ] += ( rbin ) * modWgt;
    }

#else
    for(xs  = VL_MAX (- W,       - xi) ;
        xs <= VL_MIN (+ W, w - 1 - xi) ; ++xs) {


      double dx = (double)(xi + xs) - x;
      double dy = (double)(yi + ys) - y;
      double r2 = dx*dx + dy*dy ;
      double wgt, mod, ang, fbin ;

      /* limit to a circular window */
      if (r2 >= W*W + 0.6) continue ;

      wgt  = fast_expn (f, r2 / (2*sigmaw*sigmaw)) ;
      mod  = *(pt + xs*xo + ys*yo    ) ;
      ang  = *(pt + xs*xo + ys*yo + 1) ;
      fbin = nbins * ang / (2 * VL_PI) ;

#if defined(VL_SIFT_BILINEAR_ORIENTATIONS)
      {
        int bin = (int) vl_floor_d (fbin - 0.5) ;
        double rbin = fbin - bin - 0.5 ;
        hist [(bin + nbins) % nbins] += (1 - rbin) * mod * wgt ;
        hist [(bin + 1    ) % nbins] += (    rbin) * mod * wgt ;
      }
#else
      {
        int    bin  = vl_floor_d (fbin) ;
        bin = vl_floor_d (nbins * ang / (2*VL_PI)) ;
        hist [(bin) % nbins] += mod * wgt ;
      }
#endif
  } /* for xs */
#endif

  } /* for ys */

  /* smooth histogram */
  for (iter = 0; iter < 6; iter ++) {
    double prev  = hist [nbins - 1] ;
    double first = hist [0] ;
    int i ;
    for (i = 0; i < nbins - 1; i++) {
      double newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
      prev = hist[i] ;
      hist[i] = newh ;
    }
    hist[i] = (prev + hist[i] + first) / 3.0 ;
  }

  /* find the histogram maximum */
  maxh = 0 ;
  for (i = 0 ; i < nbins ; ++i)
    maxh = VL_MAX (maxh, hist [i]) ;

  /* find peaks within 80% from max */
  nangles = 0 ;
  for(i = 0 ; i < nbins ; ++i) {
    double h0 = hist [i] ;
    double hm = hist [(i - 1 + nbins) % nbins] ;
    double hp = hist [(i + 1 + nbins) % nbins] ;

    /* is this a peak? */
    if (h0 > 0.8*maxh && h0 > hm && h0 > hp) {

      /* quadratic interpolation */
      double di = - 0.5 * (hp - hm) / (hp + hm - 2 * h0) ;
      double th = 2 * VL_PI * (i + di + 0.5) / nbins ;
      angles [ nangles++ ] = th ;
      if( nangles == 4 )
        goto enough_angles ;
    }
  }
 enough_angles:
  return nangles ;
}


/** ------------------------------------------------------------------
 ** @internal
 ** @brief Normalizes in norm L_2 a descriptor
 ** @param begin begin of histogram.
 ** @param end   end of histogram.
 **/

VL_INLINE vl_sift_pix
normalize_histogram
(vl_sift_pix *begin, vl_sift_pix *end)
{
  vl_sift_pix* iter ;
  vl_sift_pix  norm = 0.0 ;

  for (iter = begin ; iter != end ; ++ iter)
    norm += (*iter) * (*iter) ;

  norm = vl_fast_sqrt_f (norm) + VL_EPSILON_F ;

  for (iter = begin; iter != end ; ++ iter)
    *iter /= norm ;

  return norm;
}

/** ------------------------------------------------------------------
 ** @brief Run the SIFT descriptor on raw data
 **
 ** @param f        SIFT filter.
 ** @param grad     image gradients.
 ** @param descr    SIFT descriptor (output).
 ** @param width    image width.
 ** @param height   image height.
 ** @param x        keypoint x coordinate.
 ** @param y        keypoint y coordinate.
 ** @param sigma    keypoint scale.
 ** @param angle0   keypoint orientation.
 **
 ** The function runs the SIFT descriptor on raw data. Here @a image
 ** is a 2 x @a width x @a height array (by convention, the memory
 ** layout is a s such the first index is the fastest varying
 ** one). The first @a width x @a height layer of the array contains
 ** the gradient magnitude and the second the gradient angle (in
 ** radians, between 0 and @f$ 2\pi @f$). @a x, @a y and @a sigma give
 ** the keypoint center and scale respectively.
 **
 ** In order to be equivalent to a standard SIFT descriptor the image
 ** gradient must be computed at a smoothing level equal to the scale
 ** of the keypoint. In practice, the actual SIFT algorithm makes the
 ** following additional approximation, which influence the result:
 **
 ** - Scale is discretized in @c S levels.
 ** - The image is downsampled once for each octave (if you do this,
 **   the parameters @a x, @a y and @a sigma must be
 **   scaled too).
 **/

VL_EXPORT
void
vl_sift_calc_raw_descriptor (VlSiftFilt const *f,
                             vl_sift_pix const* grad,
                             vl_sift_pix *descr,
                             int width, int height,
                             double x, double y,
                             double sigma,
                             double angle0)
{
  double const magnif = f-> magnif ;

  int          w      = width ;
  int          h      = height ;
  int const    xo     = 2 ;         /* x-stride */
  int const    yo     = 2 * w ;     /* y-stride */

  int          xi     = (int) (x + 0.5) ;
  int          yi     = (int) (y + 0.5) ;

  double const st0    = sin (angle0) ;
  double const ct0    = cos (angle0) ;
  double const SBP    = magnif * sigma + VL_EPSILON_D ;
  int    const W      = floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  int const binto = 1 ;          /* bin theta-stride */
  int const binyo = NBO * NBP ;  /* bin y-stride */
  int const binxo = NBO ;        /* bin x-stride */

  int bin, dxi, dyi ;
  vl_sift_pix const *pt ;
  vl_sift_pix       *dpt ;

  /* check bounds */
  if(xi    <  0               ||
     xi    >= w               ||
     yi    <  0               ||
     yi    >= h -    1        )
    return ;

  /* clear descriptor */
  memset (descr, 0, sizeof(vl_sift_pix) * NBO*NBP*NBP) ;

  /* Center the scale space and the descriptor on the current keypoint.
   * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
   */
  pt  = grad + xi*xo + yi*yo ;
  dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo ;

#undef atd
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   */
  for(dyi =  VL_MAX(- W,   - yi   ) ;
      dyi <= VL_MIN(+ W, h - yi -1) ; ++ dyi) {

    for(dxi =  VL_MAX(- W,   - xi   ) ;
        dxi <= VL_MIN(+ W, w - xi -1) ; ++ dxi) {

      /* retrieve */
      vl_sift_pix mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      vl_sift_pix angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
      vl_sift_pix theta = vl_mod_2pi_f (angle - angle0) ;

      /* fractional displacement */
      vl_sift_pix dx = xi + dxi - x;
      vl_sift_pix dy = yi + dyi - y;

      /* get the displacement normalized w.r.t. the keypoint
         orientation and extension */
      vl_sift_pix nx = ( ct0 * dx + st0 * dy) / SBP ;
      vl_sift_pix ny = (-st0 * dx + ct0 * dy) / SBP ;
      vl_sift_pix nt = NBO * theta / (2 * VL_PI) ;

      /* Get the Gaussian weight of the sample. The Gaussian window
       * has a standard deviation equal to NBP/2. Note that dx and dy
       * are in the normalized frame, so that -NBP/2 <= dx <=
       * NBP/2. */
      vl_sift_pix const wsigma = f->windowSize ;

      vl_sift_pix win = fast_expn
        (f, (nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;

      /* The sample will be distributed in 8 adjacent bins.
         We start from the ``lower-left'' bin. */
      int         binx = (int)vl_floor_f (nx - 0.5) ;
      int         biny = (int)vl_floor_f (ny - 0.5) ;
      int         bint = (int)vl_floor_f (nt) ;
      vl_sift_pix rbinx = nx - (binx + 0.5) ;
      vl_sift_pix rbiny = ny - (biny + 0.5) ;
      vl_sift_pix rbint = nt - bint ;
      int         dbinx ;
      int         dbiny ;
      int         dbint ;

      /* Distribute the current sample into the 8 adjacent bins*/
      for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
        for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
          for(dbint = 0 ; dbint < 2 ; ++dbint) {

            if (binx + dbinx >= - (NBP/2) &&
                binx + dbinx <    (NBP/2) &&
                biny + dbiny >= - (NBP/2) &&
                biny + dbiny <    (NBP/2) ) {
              vl_sift_pix weight = win
                * mod
                * vl_abs_f (1 - dbinx - rbinx)
                * vl_abs_f (1 - dbiny - rbiny)
                * vl_abs_f (1 - dbint - rbint) ;

              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            }
          }
        }
      }
    }
  }

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if(1) {

    /* normalize L2 norm */
    vl_sift_pix norm = normalize_histogram (descr, descr + NBO*NBP*NBP) ;

    /*
       Set the descriptor to zero if it is lower than our
       norm_threshold.  We divide by the number of samples in the
       descriptor region because the Gaussian window used in the
       calculation of the descriptor is not normalized.
     */
    int numSamples =
      (VL_MIN(W, w - xi -1) - VL_MAX(-W, - xi) + 1) *
      (VL_MIN(W, h - yi -1) - VL_MAX(-W, - yi) + 1) ;

    if(f-> norm_thresh && norm < f-> norm_thresh * numSamples) {
        for(bin = 0; bin < NBO*NBP*NBP ; ++ bin)
            descr [bin] = 0;
    }
    else {
      /* truncate at 0.2. */
      for(bin = 0; bin < NBO*NBP*NBP ; ++ bin) {
        if (descr [bin] > 0.2) descr [bin] = 0.2;
      }

      /* normalize again. */
      normalize_histogram (descr, descr + NBO*NBP*NBP) ;
    }
  }
}

/** ------------------------------------------------------------------
 ** @brief Compute the descriptor of a keypoint
 **
 ** @param f        SIFT filter.
 ** @param descr    SIFT descriptor (output)
 ** @param k        keypoint.
 ** @param angle0   keypoint direction.
 **
 ** The function computes the SIFT descriptor of the keypoint @a k of
 ** orientation @a angle0. The function fills the buffer @a descr
 ** which must be large enough to hold the descriptor.
 **
 ** The function assumes that the keypoint is on the current octave.
 ** If not, it does not do anything.
 **/

VL_EXPORT
void
vl_sift_calc_keypoint_descriptor (VlSiftFilt *f,
                                  vl_sift_pix *descr,
                                  VlSiftKeypoint const* k,
                                  double angle0)
{
  /*
     The SIFT descriptor is a three dimensional histogram of the
     position and orientation of the gradient.  There are NBP bins for
     each spatial dimension and NBO bins for the orientation dimension,
     for a total of NBP x NBP x NBO bins.

     The support of each spatial bin has an extension of SBP = 3sigma
     pixels, where sigma is the scale of the keypoint.  Thus all the
     bins together have a support SBP x NBP pixels wide. Since
     weighting and interpolation of pixel is used, the support extends
     by another half bin. Therefore, the support is a square window of
     SBP x (NBP + 1) pixels. Finally, since the patch can be
     arbitrarily rotated, we need to consider a window 2W += sqrt(2) x
     SBP x (NBP + 1) pixels wide.
  */
#if FAST_SIFT_CALC_KEYPOINTS
  double const magnif      = f-> magnif ;

  double       xper        = pow (2.0, f->o_cur) ;

  int          w           = f-> octave_width ;
  int          h           = f-> octave_height ;
  int const    xo          = 2 ;         /* x-stride */
  int const    yo          = 2 * w ;     /* y-stride */
  int const    so          = 2 * w * h ; /* s-stride */
  double       x           = k-> x     / xper ;
  double       y           = k-> y     / xper ;
  double       sigma       = k-> sigma / xper ;

  int          xi          = (int) (x + 0.5) ;
  int          yi          = (int) (y + 0.5) ;
  int          si          = k-> is ;

  double const st0         = sin (angle0) ;
  double const ct0         = cos (angle0) ;
  double const SBP         = magnif * sigma + VL_EPSILON_D ;
  int    const W           = floor
  (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  int const binto = 1 ;          /* bin theta-stride */
  int const binyo = NBO * NBP ;  /* bin y-stride */
  int const binxo = NBO ;        /* bin x-stride */

  int bin, dxi, dyi ;
  vl_sift_pix const *pt ;
  vl_sift_pix       *dpt ;

  /* check bounds */
  if(k->o  != f->o_cur        ||
      xi    <  0               ||
      xi    >= w               ||
      yi    <  0               ||
      yi    >= h -    1        ||
      si    <  f->s_min + 1    ||
      si    >  f->s_max - 2     )
      return ;

  /* VL_PRINTF("W = %d ; magnif = %g ; SBP = %g\n", W,magnif,SBP) ; */

  /* clear descriptor */
  memset (descr, 0, sizeof(vl_sift_pix) * NBO*NBP*NBP) ;

  /* Center the scale space and the descriptor on the current keypoint.
   * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
   */
  pt  = f->grad + xi*xo + yi*yo + (si - f->s_min - 1)*so ;
  dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo ;

#undef atd
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   */
  float const wsigma             = f->windowSize ;
  float const invSigma2          = 1.0 / (2.0 * wsigma * wsigma); // No diff if placed below
  _Data const negInvSigmaFactor = _Set(-invSigma2);
   vl_sift_pix const angle0f     = (vl_sift_pix) angle0;
  _Data const vAngle0           = _Set( angle0 );
   double const ntFactor         = NBO / (2 * VL_PI);
  _Data const vntFactor         = _Set(ntFactor);
  _Data const vHalf             = _Set(0.5f);
  _Data const vOne              = _Set(1.f);
  _Data const vZero             = _Set(0.f);
  _DataI const vIOne             = _SetI(1);
  _Data const vBinyo            = _Set(binyo);
  _Data const vBinxo            = _Set(binxo);
  _DataI const vNBP_2            = _SetI(NBP/2);
  _DataI const vNBPN_2_1           = _SetI(-NBP/2-1);
  float const nxInc = ct0 / SBP;
  float const nyInc = -st0 / SBP;
  _Data const vNxd = _Set(nxInc*((float) GROUP_SIZE));
  _Data const vNyd = _Set(nyInc*((float) GROUP_SIZE));

  for(dyi =  VL_MAX (- W, 1 - yi    ) ;
       dyi <= VL_MIN (+ W, h - yi - 2) ; ++ dyi) {

    dxi = VL_MAX( -W, 1 - xi );
    vl_sift_pix const * __restrict pMA = pt + dxi * xo + dyi * yo;
    vl_sift_pix const dx = xi + dxi - x;
    vl_sift_pix const dy = yi + dyi - y;
    float nx = ( ct0 * dx + st0 * dy) / SBP ;
    float ny = (-st0 * dx + ct0 * dy) / SBP ;

    int const dxiMin = VL_MIN( +W, w - xi - 2 );
    int const count = dxiMin - dxi + 1;

    assert( sizeof( float ) == sizeof( vl_sift_pix ) );

    /* Calculate numGroups GROUP_SIZE pixels at a time. */
    int const numGroups = count / GROUP_SIZE;

    _Data vNx_half = _Sub(_SetNDeltas(nx, nxInc), vHalf);
    _Data vNy_half = _Sub(_SetNDeltas(ny, nyInc), vHalf);

    for (
       int i = 0; i != numGroups;
       ++i,
       pMA += GROUP_SIZE*xo, dxi += GROUP_SIZE, vNx_half = _Add(vNx_half, vNxd), vNy_half = _Add(vNy_half, vNyd)
    ) {
      _DataI const vIBinx        = _TruncateIF(Floor(vNx_half));
      _DataI const vINextBinx        = _AddI(vIBinx, vIOne);

      _DataI vXBounded = _AndI(_CmpGTI(vIBinx, vNBPN_2_1), _CmpLTI(vIBinx, vNBP_2));
      _DataI vXNextBounded = _AndI(_CmpGTI(vINextBinx, vNBPN_2_1), _CmpLTI(vINextBinx, vNBP_2));

      /* Immediately reject unbounded groups horizontally */
      if (AllZerosI(_OrI(vXBounded, vXNextBounded)))
        continue;

      _DataI const vIBiny        = _TruncateIF(Floor(vNy_half));
      _DataI const vINextBiny        = _AddI(vIBiny, vIOne);

      _DataI vYBounded = _AndI(_CmpGTI(vIBiny, vNBPN_2_1), _CmpLTI(vIBiny, vNBP_2));
      _DataI vYNextBounded = _AndI(_CmpGTI(vINextBiny, vNBPN_2_1), _CmpLTI(vINextBiny, vNBP_2));

      /* Immediately reject unbounded groups vertically */
      if (AllZerosI(_OrI(vYBounded, vYNextBounded)))
        continue;

      /* pMA of the form: m0, a0, m1, a1... */
      _Data const vModsAngles1  = _Load(pMA);
      _Data const vNx = _Add(vNx_half, vHalf);
      _Data const vNy = _Add(vNy_half, vHalf);
      _Data const vBinx         = _ConvertFI(vIBinx);
      _Data const vBiny         = _ConvertFI(vIBiny);
      _Data const vModsAngles2  = _Load(pMA+GROUP_SIZE);
      _Data const vMods         = _Evens(vModsAngles1, vModsAngles2);
      _Data const vAngles       = _Odds(vModsAngles1, vModsAngles2);
      _Data const vNts          = _Mul( Mod2PILimited(_Sub(vAngles, vAngle0)), vntFactor);

      /* Get the Gaussian weight of the sample. The Gaussian window
       * has a standard deviation equal to NBP/2. Note that dx and dy
       * are in the normalized frame, so that -NBP/2 <= dx <=
       * NBP/2. */
      _Data const vNorm         = _Mul(
                                      _Add( _Mul(vNx, vNx),_Mul(vNy, vNy) ),
                                      negInvSigmaFactor
                                   );
      _Data const vWinMod       = _Mul(FastExp(vNorm), vMods) ;

      _DataI const vIBint       = _TruncateIF(vNts);
      _Data const vBint         = _ConvertFI(vIBint); // All cases
      _Data const vRbinx        = _Sub(vNx, _Add(vBinx, vHalf)); // xbounded and (yBounded or yNextBounded)
      _Data const vRbiny        = _Sub(vNy, _Add(vBiny, vHalf));  // yBounded or yNextBounded
      _Data const vRbint        = _Sub(vNts, vBint); // All cases

      _Data const vRbint1       = _Mul(vWinMod, FastAbs(_Sub(vOne, vRbint))); // All cases
      _Data const vRbint2       = _Mul(vWinMod, FastAbs(_Sub(vZero, vRbint))); // All cases

      _Data const vRbint3       = FastAbs(_Sub(vOne, vRbiny));  // yBounded
      _Data const vRbint4       = FastAbs(_Sub(vZero, vRbiny)); // yNextBounded
      _Data const vRbint5       = FastAbs(_Sub(vOne, vRbinx));  // xBounded
      _Data const vRbint6       = FastAbs(_Sub(vZero, vRbinx)); // xNextBounded

      _Data const vRbint53      = _Mul(vRbint5, vRbint3);       // xBounded and yBounded
      _Data const v531          = _Mul(vRbint53, vRbint1);
      _Data const v532          = _Mul(vRbint53, vRbint2);

      _Data const vRbint54      = _Mul(vRbint5, vRbint4);       // xBounded and yNextBounded
      _Data const v541          = _Mul(vRbint54, vRbint1);
      _Data const v542          = _Mul(vRbint54, vRbint2);

      _Data const vRbint63      = _Mul(vRbint6, vRbint3);       // xNextBounded and yBounded
      _Data const v631          = _Mul(vRbint63, vRbint1);
      _Data const v632          = _Mul(vRbint63, vRbint2);

      _Data const vRbint64      = _Mul(vRbint6, vRbint4);       // nNextBounded and yNextBounded
      _Data const v641          = _Mul(vRbint64, vRbint1);
      _Data const v642          = _Mul(vRbint64, vRbint2);

      /* ( bint % NBO ) * binto (binto always 1) */
      _DataI const vRowColOffset = _AddI(
                                      _ShiftLI(vIBiny, 5), /* row offset binyo == 32 */
                                      _ShiftLI(vIBinx, 3)  /* col offset binxo == 8 */
                                    );

      _DataI const vDepthOffset  = _AddI(
                                      FastMod8(vIBint),
                                      vRowColOffset
                                   );
      /* ( (bint+1) % NBO ) * binto; (binto always 1) */
      _DataI const vDepthOffset2 = _AddI(
                                      FastMod8(_AddI(vIBint, vIOne)),
                                      vRowColOffset
                                     );

      for (int j = 0; j != GROUP_SIZE; ++j) {
        /* The sample will be distributed in 8 adjacent bins.
         * We start from the ``lower-left'' bin. */
        int const xBounded       = _AsArrayI(vXBounded, j);
        int const xNextBounded   = _AsArrayI(vXNextBounded, j);
        int const yBounded       = _AsArrayI(vYBounded, j);
        int const yNextBounded   = _AsArrayI(vYNextBounded, j);
      
        float* __restrict p1
                                 = dpt + _AsArrayI(vDepthOffset,j);
        float* __restrict p2
                                 = dpt + _AsArrayI(vDepthOffset2,j);
        float* __restrict p3     = p1 + 32; /* binyo == 32 */
        float* __restrict p4     = p2 + 32;
      
        /* Distribute the current sample into the 8 adjacent bins*/
      
        /* Not faster to avoid the branches. */
        if (xBounded) {
            if (yBounded) {
              *p1 += _AsArrayF(v531,j); /* xBound < NBP && yBound < NBP */
              *p2 += _AsArrayF(v532,j); /* xBound < NBP && yBound < NBP */
            }

            if (yNextBounded) {
              *p3 += _AsArrayF(v541,j); /* xBound < NBP && (yBound+1) < NBP */
              *p4 += _AsArrayF(v542,j); /* xBound < NBP && (yBound+1) < NBP */
            }
          }
      
        if (xNextBounded) {
         p1 += binxo;
         p2 += binxo;

         if (yBounded) {
             *p1 += _AsArrayF(v631,j) ; /* (xBound+1) < NBP && yBound < NBP */
             *p2 += _AsArrayF(v632,j) ; /* (xBound+1) < NBP && yBound < NBP */
         }

         if (yNextBounded) {
           p3 += binxo;
           p4 += binxo;
           *p3 += _AsArrayF(v641,j) ; /* (xBound+1) < NBP && (yBound+1) < NBP */
           *p4 += _AsArrayF(v642,j) ; /* (xBound+1) < NBP && (yBound+1) < NBP */
         }
       }
      }
    }

    nx = _AsArrayF(vNx_half, 0) + 0.5f;
    ny = _AsArrayF(vNy_half, 0) + 0.5f;

    /* Perform the processing on the remaining elements. */
    for( ; dxi <= dxiMin ;++dxi, pMA += xo, nx += nxInc, ny += nyInc) {
      /* retrieve */
      vl_sift_pix const mod   = pMA[0] ;
      vl_sift_pix const angle = pMA[1] ;
      vl_sift_pix const theta = Mod2PILimitedS(angle - angle0) ;

      /* get the displacement normalized w.r.t. the keypoint
      orientation and extension */
      vl_sift_pix const nt = theta * ntFactor ;

      /* Get the Gaussian weight of the sample. The Gaussian window
       * has a standard deviation equal to NBP/2. Note that dx and dy
       * are in the normalized frame, so that -NBP/2 <= dx <=
       * NBP/2. */
      vl_sift_pix const win =  fastExp3S( -( nx * nx + ny * ny ) * invSigma2 );

      vl_sift_pix const winMod = win * mod;

      /* The sample will be distributed in 8 adjacent bins.
      We start from the ``lower-left'' bin. */
      int         binx = (int)vl_floor_f (nx - 0.5f) ;
      int         biny = (int)vl_floor_f (ny - 0.5f) ;
      int         bint = (int)vl_floor_f (nt) ;
      vl_sift_pix rbinx = nx - (binx + 0.5f) ;
      vl_sift_pix rbiny = ny - (biny + 0.5f) ;
      vl_sift_pix rbint = nt - bint ;
      int         dbinx ;
      int         dbiny ;
      int         dbint ;
      vl_sift_pix const rbintFactor1 = vl_abs_f( 1.f - rbint );
      vl_sift_pix const rbintFactor2 = vl_abs_f( -rbint );
      vl_sift_pix* dpt_bint_binto = dpt + ( bint % NBO ) * binto;
      vl_sift_pix* dpt_bint_binto2 = dpt + ( (bint+1) % NBO ) * binto;
      vl_sift_pix const rbint1 = winMod * rbintFactor1;
      vl_sift_pix const rbint2 = winMod * rbintFactor2;

      /* Distribute the current sample into the 8 adjacent bins*/
      int xBound = binx + 0 + NBP / 2;
      int binx_dbinx_binxo = ( binx + 0 ) * binxo;
      for(dbinx = 0 ; dbinx < 2 ; ++dbinx, ++xBound, binx_dbinx_binxo += binxo) {
        if (( unsigned ) xBound < NBP) {
          int yBound = biny + 0 + NBP / 2;
          if (( unsigned ) yBound < NBP) {
            int addressXY = ( biny + 0 ) * binyo + binx_dbinx_binxo;
            float const weightFactorX = vl_abs_f( 1 - dbinx - rbinx );
            vl_sift_pix const weightFactorXY = weightFactorX * vl_abs_f( 1 - rbiny );

            dpt_bint_binto[ addressXY ] += weightFactorXY * rbint1;
            dpt_bint_binto2[ addressXY ] += weightFactorXY * rbint2;

            ++yBound;
            if (( unsigned ) ( yBound ) < NBP) {
              addressXY += binyo;
              vl_sift_pix const weightFactorXY = weightFactorX * vl_abs_f( -rbiny );

              dpt_bint_binto[ addressXY ] += weightFactorXY * rbint1;
              dpt_bint_binto2[ addressXY ] += weightFactorXY * rbint2;
            }
          } else {
            if (( unsigned ) ( yBound + 1 ) < NBP) {
              int const addressXY = ( biny + 1 ) * binyo + binx_dbinx_binxo;
              float const weightFactorX = vl_abs_f( 1 - dbinx - rbinx );
              vl_sift_pix const weightFactorXY = weightFactorX * vl_abs_f( -rbiny );

              dpt_bint_binto[ addressXY ] += weightFactorXY * rbint1;
              dpt_bint_binto2[ addressXY ] += weightFactorXY * rbint2;
            }
          }
        }
      }
    }
  }

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if(1) {

      /* Normalize the histogram to L2 unit length. */
      vl_sift_pix norm = normalize_histogram (descr, descr + NBO*NBP*NBP) ;

      /* Set the descriptor to zero if it is lower than our norm_threshold */
      if(f-> norm_thresh && norm < f-> norm_thresh) {
          for(bin = 0; bin < NBO*NBP*NBP ; ++ bin)
              descr [bin] = 0;
      }
      else {

          /* Truncate at 0.2. */
          for(bin = 0; bin < NBO*NBP*NBP ; ++ bin) {
              if (descr [bin] > 0.2) descr [bin] = 0.2;
          }

          /* Normalize again. */
          normalize_histogram (descr, descr + NBO*NBP*NBP) ;
      }
  }
#else
  double const magnif      = f-> magnif ;

  double       xper        = pow (2.0, f->o_cur) ;

  int          w           = f-> octave_width ;
  int          h           = f-> octave_height ;
  int const    xo          = 2 ;         /* x-stride */
  int const    yo          = 2 * w ;     /* y-stride */
  int const    so          = 2 * w * h ; /* s-stride */
  double       x           = k-> x     / xper ;
  double       y           = k-> y     / xper ;
  double       sigma       = k-> sigma / xper ;

  int          xi          = (int) (x + 0.5) ;
  int          yi          = (int) (y + 0.5) ;
  int          si          = k-> is ;

  double const st0         = sin (angle0) ;
  double const ct0         = cos (angle0) ;
  double const SBP         = magnif * sigma + VL_EPSILON_D ;
  int    const W           = floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  int const binto = 1 ;          /* bin theta-stride */
  int const binyo = NBO * NBP ;  /* bin y-stride */
  int const binxo = NBO ;        /* bin x-stride */

  int bin, dxi, dyi ;
  vl_sift_pix const *pt ;
  vl_sift_pix       *dpt ;

  /* check bounds */
  if(k->o  != f->o_cur        ||
     xi    <  0               ||
     xi    >= w               ||
     yi    <  0               ||
     yi    >= h -    1        ||
     si    <  f->s_min + 1    ||
     si    >  f->s_max - 2     )
    return ;

  /* VL_PRINTF("W = %d ; magnif = %g ; SBP = %g\n", W,magnif,SBP) ; */

  /* clear descriptor */
  memset (descr, 0, sizeof(vl_sift_pix) * NBO*NBP*NBP) ;

  /* Center the scale space and the descriptor on the current keypoint.
   * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
   */
  pt  = f->grad + xi*xo + yi*yo + (si - f->s_min - 1)*so ;
  dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo ;

#undef atd
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   */
  for(dyi =  VL_MAX (- W, 1 - yi    ) ;
      dyi <= VL_MIN (+ W, h - yi - 2) ; ++ dyi) {

    for(dxi =  VL_MAX (- W, 1 - xi    ) ;
        dxi <= VL_MIN (+ W, w - xi - 2) ; ++ dxi) {

      /* retrieve */
      vl_sift_pix mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      vl_sift_pix angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
      vl_sift_pix theta = vl_mod_2pi_f (angle - angle0) ;

      /* fractional displacement */
      vl_sift_pix dx = xi + dxi - x;
      vl_sift_pix dy = yi + dyi - y;

      /* get the displacement normalized w.r.t. the keypoint
         orientation and extension */
      vl_sift_pix nx = ( ct0 * dx + st0 * dy) / SBP ;
      vl_sift_pix ny = (-st0 * dx + ct0 * dy) / SBP ;
      vl_sift_pix nt = NBO * theta / (2 * VL_PI) ;

      /* Get the Gaussian weight of the sample. The Gaussian window
       * has a standard deviation equal to NBP/2. Note that dx and dy
       * are in the normalized frame, so that -NBP/2 <= dx <=
       * NBP/2. */
      vl_sift_pix const wsigma = f->windowSize ;
      vl_sift_pix win = fast_expn
        (f, (nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;

      /* The sample will be distributed in 8 adjacent bins.
         We start from the ``lower-left'' bin. */
      int         binx = (int)vl_floor_f (nx - 0.5) ;
      int         biny = (int)vl_floor_f (ny - 0.5) ;
      int         bint = (int)vl_floor_f (nt) ;
      vl_sift_pix rbinx = nx - (binx + 0.5) ;
      vl_sift_pix rbiny = ny - (biny + 0.5) ;
      vl_sift_pix rbint = nt - bint ;
      int         dbinx ;
      int         dbiny ;
      int         dbint ;

      /* Distribute the current sample into the 8 adjacent bins*/
      for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
        for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
          for(dbint = 0 ; dbint < 2 ; ++dbint) {

            if (binx + dbinx >= - (NBP/2) &&
                binx + dbinx <    (NBP/2) &&
                biny + dbiny >= - (NBP/2) &&
                biny + dbiny <    (NBP/2) ) {
              vl_sift_pix weight = win
                * mod
                * vl_abs_f (1 - dbinx - rbinx)
                * vl_abs_f (1 - dbiny - rbiny)
                * vl_abs_f (1 - dbint - rbint) ; // rbintFactor1 at dbint = 0 and rbinFactor2 at dbing == 1

              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            }
          }
        }
      }
    }
  }

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if(1) {

    /* Normalize the histogram to L2 unit length. */
    vl_sift_pix norm = normalize_histogram (descr, descr + NBO*NBP*NBP) ;

    /* Set the descriptor to zero if it is lower than our norm_threshold */
    if(f-> norm_thresh && norm < f-> norm_thresh) {
        for(bin = 0; bin < NBO*NBP*NBP ; ++ bin)
            descr [bin] = 0;
    }
    else {

      /* Truncate at 0.2. */
      for(bin = 0; bin < NBO*NBP*NBP ; ++ bin) {
        if (descr [bin] > 0.2) descr [bin] = 0.2;
      }

      /* Normalize again. */
      normalize_histogram (descr, descr + NBO*NBP*NBP) ;
    }
  }
#endif
}

/** ------------------------------------------------------------------
 ** @brief Initialize a keypoint from its position and scale
 **
 ** @param f     SIFT filter.
 ** @param k     SIFT keypoint (output).
 ** @param x     x coordinate of the keypoint center.
 ** @param y     y coordinate of the keypoint center.
 ** @param sigma keypoint scale.
 **
 ** The function initializes a keypoint structure @a k from
 ** the location @a x
 ** and @a y and the scale @a sigma of the keypoint. The keypoint structure
 ** maps the keypoint to an octave and scale level of the discretized
 ** Gaussian scale space, which is required for instance to compute the
 ** keypoint SIFT descriptor.
 **
 ** @par Algorithm
 **
 ** The formula linking the keypoint scale sigma to the octave and
 ** scale indexes is
 **
 ** @f[ \sigma(o,s) = \sigma_0 2^{o+s/S} @f]
 **
 ** In addition to the scale index @e s (which can be fractional due
 ** to scale interpolation) a keypoint has an integer scale index @e
 ** is too (which is the index of the scale level where it was
 ** detected in the DoG scale space). We have the constraints (@ref
 ** sift-tech-detector see also the "SIFT detector"):
 **
 ** - @e o is integer in the range @f$ [o_\mathrm{min},
 **   o_{\mathrm{min}}+O-1] @f$.
 ** - @e is is integer in the range @f$ [s_\mathrm{min}+1,
 **   s_\mathrm{max}-2] @f$.  This depends on how the scale is
 **   determined during detection, and must be so here because
 **   gradients are computed only for this range of scale levels
 **   and are required for the calculation of the SIFT descriptor.
 ** - @f$ |s - is| < 0.5 @f$ for detected keypoints in most cases due
 **   to the interpolation technique used during detection. However
 **   this is not necessary.
 **
 ** Thus octave o represents scales @f$ \{ \sigma(o, s) : s \in
 ** [s_\mathrm{min}+1-.5, s_\mathrm{max}-2+.5] \} @f$. Note that some
 ** scales may be represented more than once. For each scale, we
 ** select the largest possible octave that contains it, i.e.
 **
 ** @f[
 **  o(\sigma)
 **  = \max \{ o \in \mathbb{Z} :
 **    \sigma_0 2^{\frac{s_\mathrm{min}+1-.5}{S}} \leq \sigma \}
 **  = \mathrm{floor}\,\left[
 **    \log_2(\sigma / \sigma_0) - \frac{s_\mathrm{min}+1-.5}{S}\right]
 ** @f]
 **
 ** and then
 **
 ** @f[
 ** s(\sigma) = S  \left[\log_2(\sigma / \sigma_0) - o(\sigma)\right],
 ** \quad
 ** is(\sigma) = \mathrm{round}\,(s(\sigma))
 ** @f]
 **
 ** In practice, both @f$ o(\sigma) @f$ and @f$ is(\sigma) @f$ are
 ** clamped to their feasible range as determined by the SIFT filter
 ** parameters.
 **/

VL_EXPORT
void
vl_sift_keypoint_init (VlSiftFilt const *f,
                       VlSiftKeypoint *k,
                       double x,
                       double y,
                       double sigma)
{
  int    o, ix, iy, is ;
  double s, phi, xper ;

  phi = log2 ((sigma + VL_EPSILON_D) / f->sigma0) ;
  o   = (int)vl_floor_d (phi -  ((double) f->s_min + 0.5) / f->S) ;
  o   = VL_MIN (o, f->o_min + f->O - 1) ;
  o   = VL_MAX (o, f->o_min           ) ;
  s   = f->S * (phi - o) ;

  is  = (int)(s + 0.5) ;
  is  = VL_MIN(is, f->s_max - 2) ;
  is  = VL_MAX(is, f->s_min + 1) ;

  xper = pow (2.0, o) ;
  ix   = (int)(x / xper + 0.5) ;
  iy   = (int)(y / xper + 0.5) ;

  k -> o  = o ;

  k -> ix = ix ;
  k -> iy = iy ;
  k -> is = is ;

  k -> x = x ;
  k -> y = y ;
  k -> s = s ;

  k->sigma = sigma ;
}
