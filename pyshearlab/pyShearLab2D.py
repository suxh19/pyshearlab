"""
This module contains all relevant transform functions for the 2D case provided
by the ShearLab3D toolbox from MATLAB such as the forward and inverse
transform and the construction of a shearlet system.

All these functions were originally written by Rafael Reisenhofer and are
published in the ShearLab3Dv11 toolbox on http://www.shearlet.org.

Stefan Loock, February 2, 2017 [sloock@gwdg.de]
"""

from __future__ import division
import numpy as np
from pyshearlab.pySLFilters import *
from pyshearlab.pySLUtilities import *

def SLgetShearletSystem2D(useGPU, rows, cols, nScales, shearLevels=None, full=0, directionalFilter=None, quadratureMirrorFilter=None):
    """
    Compute a 2D shearlet system.

    Usage:

        shearletSystem = SLgetShearletSystem2D(useGPU, rows, cols,
                        nScales)
        shearletSystem = SLgetShearletSystem2D(useGPU, rows, cols,
                        nScales, shearLevels)
        shearletSystem = SLgetShearletSystem2D(useGPU, rows, cols,
                         nScales, shearLevels, full)
        shearletSystem = SLgetShearletSystem2D(useGPU, rows, cols,
                        nScales, shearLevels, full,
                        directionalFilter)
        shearletSystem = SLgetShearletSystem2D(useGPU, rows, cols,
                        nScales, shearLevels, full,
                        directionalFilter, quadratureMirrorFilter)

    Input:

        useGPU: Logical value, determines if the
                shearlet system is constructed on the GPU.
        rows:    Number of rows.
        cols:    Number of columns.
        nScales: Number of scales of the desired shearlet system.
                 Has to be >= 1.
        shearLevels: A 1xnScales sized array, specifying the level of
                     shearing occuring on each scale. Each entry of
                     shearLevels has to be >= 0. A shear level of K
                     means that the generating shearlet is sheared
                     2^K times in each direction for each cone.
                     For example: If nScales = 3 and
                     shearLevels = [1 1 2], the shearlet system will
                     contain
                     (2*(2*2^1+1))+(2*(2*2^1+1))+(2*(2*2^2+1))=38
                     shearlets (omitting the lowpass shearlet and
                     translation). Note that it is recommended not
                     to use the full shearlet system but to omit
                     shearlets lying on the border of the second
                     cone as they are only slightly different from
                     those on the border of the first cone. The
                     default value is ceil((1:nScales)/2).
        full:       Logical value that determines whether a full
                    shearlet system is computed or if shearlets
                    lying on the border of the second cone are
                    omitted. The default and recommended value
                    is 0.
        directionalFilter: A 2D directional filter that serves as the
                    basis of the directional 'component' of the
                    shearlets.
                    The default choice is
                        modulate2(dfilters('dmaxflat4','d'),'c').
                    For small sized inputs or very large systems, the
                    default directional filter might be too large. In
                    this case, it is recommended to use
                             modulate2(dfilters('cd','d'),'c').
        quadratureMirrorFilter: A 1D quadrature mirror filter
                    defining the wavelet 'component' of the
                    shearlets. The default choice is
                    [0.0104933261758410,-0.0263483047033631,-0.0517766952966370,
                     0.276348304703363,0.582566738241592,0.276348304703363,
                     -0.0517766952966369,-0.0263483047033631,0.0104933261758408].

                    Other QMF filters can be genereted with
                    MakeONFilter.

    Output:

        shearletSystem: A structure containing the specified shearlet
                        system.
        ["shearlets"]: A X x Y x N array of N 2D shearlets in the
                        frequency domain where X and Y denote the
                        size of each shearlet. To get the i-th
                        shearlet in the time domain, use
        fftshift(ifft2(ifftshift(shearletSystem.shearlets(:,:,i)))).
                    Each Shearlet is centered at floor([X Y]/2)+1.
        ["size"]:       The size of each shearlet in the system.
        ["shearLevels"]: The respective input argument is stored
                        here.
        ["full"]:       The respective input argument is stored here.
        ["nShearlets"]: Number of shearlets in the
                        shearletSystem["shearlets"] array. This
                        number also describes the redundancy of
                        the system.
        ["shearletdIdxs"]: A Nx3 array, specifying each shearlet in
                            the system in the format
                            [cone scale shearing] where N denotes
                            the number of shearlets. The vertical
                            cone in the time domain is indexed by
                            1 while the horizontal cone is indexed
                            by 2.
                            Note that the values for scale and
                            shearing are limited by specified
                            number of scales and shaer levels. The
                            lowpass shearlet is indexed by [0 0 0].
        ["dualFrameWeights"]: A XxY matrix containing the absolute
                            and squared sum over all shearlets
                            stored in shearletSystem.shearlets.
                            These weights are needed to compute
                            the dual frame during reconstruction.
        ["RMS"]:            A 1xnShearlets array containing the root
                            mean squares (L2-norm divided by
                            sqrt(X*Y)) of all shearlets stored in
                            shearletSystem["shearlets"]. These
                            values can be used to normalize
                            shearlet coefficients to make them
                            comparable.
        ["useGPU"]:         Logical value. Tells if the shearlet
                            system is stored on the GPU.
                            Right now this is ignored since no GPU
                            implementation is done yet.

    Example 1:
    compute a standard shearlet system of four scales

        shearletSystem = SLgetShearletSystem2D(0,512,512,4)

    Example 2:
    compute a full shearlet system of four scales

        shearletSystem = SLgetShearletSystem2D(0,512,512,4,
                                        [1 1 2 2],1)

    Example 3:
    compute a shearlet system with high shear levels for small sized data
    using a non-default directional filter.

        directionalFilter = modulate2(dfilters('cd','d'),'c')
        shearletSystem = SLgetShearletSystem2D(0,256,256,4,
                            [2 2 3 3],0,directionalFilter)



    See also: SLgetShearletIdxs2D,SLsheardec2D,SLshearrec2D,SLgetSubsystem2D
    """
    # check which args are given and set default values if necccessary
    if shearLevels is None:
        shearLevels = np.ceil(np.arange(1,nScales+1)/2).astype(int)
    if directionalFilter is None:
        h0, h1 = dfilters('dmaxflat4', 'd')
        h0 = h0 / np.sqrt(2)
        h1 = h1 / np.sqrt(2)
        directionalFilter = modulate2(h0, 'c')
    if quadratureMirrorFilter is None:
        quadratureMirrorFilter = np.array([0.0104933261758410, -0.0263483047033631,
                        -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                        0.276348304703363, -0.0517766952966369, -0.0263483047033631,
                        0.0104933261758408])
    # skipping use gpu stuff for the moment...
    preparedFilters = SLprepareFilters2D(rows,cols,nScales,shearLevels, directionalFilter,quadratureMirrorFilter)
    shearletIdxs = SLgetShearletIdxs2D(shearLevels, full)
    shearlets, RMS, dualFrameWeights = SLgetShearlets2D(preparedFilters, shearletIdxs)

    # create dictionary
    shearletSystem = {"shearlets": shearlets, "size": preparedFilters["size"],
                        "shearLevels": preparedFilters["shearLevels"],
                        "full": full, "nShearlets": shearletIdxs.shape[0],
                        "shearletIdxs": shearletIdxs, "dualFrameWeights": dualFrameWeights,
                        "RMS": RMS, "useGPU": useGPU, "isComplex": 0}
    return shearletSystem


def SLsheardec2D(X, shearletSystem):
    """
    Shearlet decomposition of 2D data.

    Usage:

        coeffs = SLsheardec2D(X,shearletSystem);

    Input:

        X:              2D data in time domain.
        shearletSystem: Structure containg a shearlet system. Such a
                        system can be computed with
                        SLgetShearletSystem2D or SLgetSubsystem2D.
    Output:

        coeffs: X x Y x N array of the same size as the
                shearletSystem["shearlets"] array. coeffs contains
                all shearlet coefficients, that is all inner products
                with the given data, of all translates of the
                shearlets in the specified system. When constructing
                shearlets with SLgetShearletSystem2D, each shearlet
                is centered in the time domain at floor(size(X)/2)+1.
                Hence, the inner product of X and the i-th
                shearlet in the time domain can be found at
                coeffs(floor(size(X,1)/2)+1,floor(size(X,2)/2)+1,i).

    Example:

        X = double(imread('barbara.jpg'))
        useGPU = 0
        shearletSystem = SLgetShearletSystem2D(useGPU,size(X,1),size(X,2),4)
        coeffs = SLsheardec2D(X,shearletSystem)

    See also: SLgetShearletSystem2D, SLgetSubsystem2D, SLshearrec2D
    """
    #skipping useGPU stuff...
    coeffs = np.zeros(shearletSystem["shearlets"].shape, dtype=complex)

    # get data in frequency domain
    Xfreq = fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(X)))

    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    for j in range(shearletSystem["nShearlets"]):
        coeffs[:,:,j] = fftlib.fftshift(fftlib.ifft2(fftlib.ifftshift(Xfreq*np.conj(shearletSystem["shearlets"][:,:,j]))))

    # probably due to rounding errors, the result may have imaginary parts with
    # small magnitude. if they are small enough, we can ignore them. otherwise
    # we report an error.
    err_bound = np.finfo(X.dtype).resolution * 1e3
    if np.max(np.abs(np.imag(coeffs))) > err_bound:
        print("Warning: magnitude in imaginary part exceeded {}.".format(err_bound))
        print("Data is probably not real-valued. Largest magnitude: " + str(np.max(np.abs(np.imag(coeffs)))))
        print("Imaginary part neglected.")

    return np.real(coeffs).astype(X.dtype)


def SLshearrec2D(coeffs, shearletSystem):
    """
    2D reconstruction of shearlet coefficients.

    Usage:

        X = SLshearrec2D(coeffs, shearletSystem)

    Input:

        coeffs:          X x Y x N array of shearlet coefficients.
        shearletSystem: Structure containing a shearlet system. This
                        should be the same system as the one
                        previously used for decomposition.

    Output:

        X: Reconstructed 2D data.

    Example:

        X = double(imread('barbara.jpg'))
        useGPU = 0
        shearletSystem = SLgetShearletSystem2D(useGPU,size(X,1),size(X,2),4)
        coeffs = SLsheardec2D(X,shearletSystem)
        Xrec = SLshearrec2D(coeffs,shearletSystem)

    See also: SLgetShearlets2D,SLsheardec2D
    """
    # skipping useGPU stuff...
    X = np.zeros((coeffs.shape[0], coeffs.shape[1]), dtype=complex)

    for j in range(shearletSystem["nShearlets"]):
        X = X + fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(coeffs[:,:,j])))*shearletSystem["shearlets"][:,:,j]

    X = fftlib.fftshift(fftlib.ifft2(fftlib.ifftshift((np.divide(X,shearletSystem["dualFrameWeights"])))))

    # probably due to rounding errors, the result may have imaginary parts with
    # small magnitude. if they are small enough, we can ignore them. otherwise
    # we report an error.
    err_bound = np.finfo(X.dtype).resolution * 1e3
    if np.max(np.abs(np.imag(X))) > err_bound:
        print("Warning: magnitude in imaginary part exceeded {}.".format(err_bound))
        print("Data is probably not real-valued. Largest magnitude: " + str(np.max(np.abs(np.imag(X)))))
        print("Imaginary part neglected.")

    return np.real(X).astype(coeffs.dtype)



def SLshearadjoint2D(coeffs, shearletSystem):
    """
    2D adjoint from shearlet coefficients.

    Usage:

        X = SLshearrec2D(coeffs, shearletSystem)

    Input:

        coeffs:          X x Y x N array of shearlet coefficients.
        shearletSystem: Structure containing a shearlet system. This
                        should be the same system as the one
                        previously used for decomposition.

    Output:

        X: Adjoint, 2D data.

    Example:

        X = double(imread('barbara.jpg'))
        useGPU = 0
        shearletSystem = SLgetShearletSystem2D(useGPU,size(X,1),size(X,2),4)
        coeffs = SLsheardec2D(X,shearletSystem)
        Xadj = SLshearadjoint2D(coeffs,shearletSystem)

        # Verify adjoint equation
        print(np.vdot(X, Xadj), np.vdot(coeffs, coeffs))

    See also: SLgetShearlets2D, SLsheardec2D, SLshearrec2D
    """
    # skipping useGPU stuff...
    X = np.zeros((coeffs.shape[0], coeffs.shape[1]), dtype=complex)

    for j in range(shearletSystem["nShearlets"]):
        X += shearletSystem["shearlets"][:,:,j] * fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(coeffs[..., j])))

    # get data in frequency domain
    Xresult = fftlib.fftshift(fftlib.ifft2(fftlib.ifftshift(X)))

    # Do not perform check for imag part here since adjoint should ignore it.

    return np.real(Xresult).astype(coeffs.dtype)


def SLshearrecadjoint2D(X, shearletSystem):
    """
    Adjoint of (pseudo-)inverse of 2D data.

    Note that this is also the (pseudo-)inverse of the adjoint.

    Usage:

        coeffs = SLshearrecadjoint2D(X, shearletSystem)

    Input:

        X             : 2D data.
        shearletSystem: Structure containing a shearlet system. This
                        should be the same system as the one
                        previously used for decomposition.

    Output:

        coeffs:          X x Y x N array of shearlet coefficients.
    """
    # skipping useGPU stuff...

    # STUFF
    Xfreq = np.divide(fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(X))), shearletSystem["dualFrameWeights"])
    coeffs = np.zeros(shearletSystem["shearlets"].shape, dtype=complex)

    for j in range(shearletSystem["nShearlets"]):
        coeffs[:,:,j] = fftlib.fftshift(fftlib.ifft2(fftlib.ifftshift(Xfreq*np.conj(shearletSystem["shearlets"][:,:,j]))))

    return np.real(coeffs).astype(X.dtype)

#
##############################################################################


##############################################################################
# High-level API with Automatic Square Padding for Non-Square Images
##############################################################################

def SLsheardecPadded2D(X, nScales, shearLevels=None, pad_mode='symmetric', 
                        full=0, directionalFilter=None, quadratureMirrorFilter=None):
    """
    Shearlet decomposition with automatic square padding for non-square images.
    
    This function solves the filter wrapping problem when processing images with
    extreme aspect ratios (e.g., 256x1024). It automatically:
    1. Pads the input to a square using symmetric (mirror) padding
    2. Creates a shearlet system for the padded size
    3. Performs decomposition
    4. Crops the coefficients back to the original image size
    
    Usage:
    
        coeffs, context = SLsheardecPadded2D(X, nScales)
        coeffs, context = SLsheardecPadded2D(X, nScales, shearLevels=[1,1,2])
        coeffs, context = SLsheardecPadded2D(X, nScales, pad_mode='reflect')
    
    Input:
    
        X:           2D data in time domain (can be non-square).
        nScales:     Number of scales for the shearlet system. Has to be >= 1.
        shearLevels: Optional. A 1xnScales array specifying shear levels per scale.
                     Default is ceil((1:nScales)/2).
        pad_mode:    Padding mode for symmetric padding. Default is 'symmetric'.
                     Use 'reflect' for edge-reflected padding.
        full:        Logical value for full shearlet system. Default is 0.
        directionalFilter: Optional 2D directional filter.
        quadratureMirrorFilter: Optional 1D QMF filter.
    
    Output:
    
        coeffs:  X x Y x N array of shearlet coefficients, where X and Y match
                 the ORIGINAL input size (not the padded size).
        context: Dictionary containing information needed for reconstruction:
                 - 'shearletSystem': The shearlet system for the padded image
                 - 'pad_info': Padding information from SLsymmetricPad2D
                 - 'original_dtype': Original data type
                 - 'coeffs_padded': Full padded coefficients (for exact reconstruction)
    
    Example:
    
        # Process a 256x1024 sinogram without filter wrapping
        coeffs, ctx = SLsheardecPadded2D(sinogram, nScales=3)
        
        # Modify coefficients (e.g., thresholding for denoising)
        coeffs_modified = coeffs * (np.abs(coeffs) > threshold)
        
        # Reconstruct
        result = SLshearrecPadded2D(coeffs_modified, ctx)
    
    See also: SLshearrecPadded2D, SLsheardec2D, SLsymmetricPad2D
    """
    # Step 1: Pad input to square using symmetric padding
    X_padded, pad_info = SLsymmetricPad2D(X, make_square=True, pad_mode=pad_mode)
    padded_rows, padded_cols = X_padded.shape
    
    # Step 2: Create shearlet system for the padded (square) image
    shearletSystem = SLgetShearletSystem2D(
        useGPU=0, 
        rows=padded_rows, 
        cols=padded_cols, 
        nScales=nScales,
        shearLevels=shearLevels,
        full=full,
        directionalFilter=directionalFilter,
        quadratureMirrorFilter=quadratureMirrorFilter
    )
    
    # Step 3: Perform decomposition on padded image
    coeffs_padded = SLsheardec2D(X_padded, shearletSystem)
    
    # Step 4: Crop coefficients back to original size (for user convenience)
    coeffs = SLcrop2D(coeffs_padded, pad_info)
    
    # Store context for reconstruction
    # IMPORTANT: We store the full padded coefficients for exact reconstruction
    context = {
        'shearletSystem': shearletSystem,
        'pad_info': pad_info,
        'original_dtype': X.dtype,
        'coeffs_padded': coeffs_padded  # Full coefficients for reconstruction
    }
    
    return coeffs, context


def SLshearrecPadded2D(coeffs, context):
    """
    Reconstruct from shearlet coefficients obtained via SLsheardecPadded2D.
    
    This function handles the padding/cropping automatically. If the input 
    coefficients match the original decomposition (not modified), exact 
    reconstruction is performed. If coefficients were modified (e.g., for 
    denoising), the modifications are applied to the central region.
    
    Usage:
    
        X_rec = SLshearrecPadded2D(coeffs, context)
    
    Input:
    
        coeffs:  X x Y x N array of shearlet coefficients (original image size).
                 These should match the size returned by SLsheardecPadded2D.
                 Can be modified (e.g., thresholded for denoising).
        context: Dictionary returned by SLsheardecPadded2D containing:
                 - 'shearletSystem': The shearlet system used for decomposition
                 - 'pad_info': Padding information
                 - 'original_dtype': Original data type
                 - 'coeffs_padded': Full padded coefficients from decomposition
    
    Output:
    
        X: Reconstructed 2D data with the same size as the original input.
    
    Example:
    
        # Full decomposition-reconstruction cycle (exact)
        coeffs, ctx = SLsheardecPadded2D(image, nScales=3)
        reconstructed = SLshearrecPadded2D(coeffs, ctx)
        
        # With coefficient modification (e.g., denoising)
        coeffs, ctx = SLsheardecPadded2D(noisy_image, nScales=3)
        coeffs_denoised = coeffs * (np.abs(coeffs) > threshold)
        denoised = SLshearrecPadded2D(coeffs_denoised, ctx)
    
    See also: SLsheardecPadded2D, SLshearrec2D, SLcrop2D
    """
    shearletSystem = context['shearletSystem']
    pad_info = context['pad_info']
    original_dtype = context['original_dtype']
    coeffs_padded_original = context['coeffs_padded']
    
    pad_top = pad_info['pad_top']
    pad_left = pad_info['pad_left']
    original_rows, original_cols = pad_info['original_shape']
    
    # Check if coefficients were modified
    # If coefficients are the same as during decomposition, use the full padded coefficients
    # Otherwise, update the central region with the modified coefficients
    coeffs_padded = coeffs_padded_original.copy()
    coeffs_padded[pad_top : pad_top + original_rows,
                  pad_left : pad_left + original_cols,
                  :] = coeffs
    
    # Perform reconstruction on the padded coefficients
    X_padded = SLshearrec2D(coeffs_padded, shearletSystem)
    
    # Crop result back to original size
    X_rec = SLcrop2D(X_padded, pad_info)
    
    return X_rec.astype(original_dtype)

