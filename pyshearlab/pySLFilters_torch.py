"""
PyTorch implementation of pyShearLab2D filters.
Translated from pySLFilters.py (NumPy version).

This module provides the same filtering functions as pySLFilters.py but using
PyTorch tensors instead of NumPy arrays, enabling GPU acceleration.

Stefan Loock (original NumPy), PyTorch translation 2024
"""

from __future__ import division
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union


def MakeONFilter(Type: str, Par: int = 1, 
                 dtype: torch.dtype = torch.float64, 
                 device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Generate Orthonormal QMF Filter for Wavelet Transform (PyTorch version).
    
    This is a translation of the original Matlab implementation of MakeONFilter.m
    from the WaveLab850 toolbox.

    Args:
        Type: Filter type. One of 'Haar', 'Beylkin', 'Coiflet', 'Daubechies',
              'Symmlet', 'Vaidyanathan', 'Battle'
        Par: Parameter for some filter types (default: 1)
        dtype: Output tensor dtype (default: torch.float64)
        device: Device for the output tensor (default: 'cpu')

    Returns:
        Orthonormal quadrature mirror filter as a 1D tensor
    """
    onFilter = None
    
    if Type == 'Haar':
        onFilter = torch.tensor([1/torch.sqrt(torch.tensor(2.0)), 
                                  1/torch.sqrt(torch.tensor(2.0))], dtype=dtype, device=device)
    elif Type == 'Beylkin':
        onFilter = torch.tensor([.099305765374, .424215360813, .699825214057,
                    .449718251149, -.110927598348, -.264497231446,
                    .026900308804, .155538731877, -.017520746267,
                    -.088543630623, .019679866044, .042916387274,
                    -.017460408696, -.014365807969, .010040411845,
                    .001484234782, -.002736031626, .000640485329], dtype=dtype, device=device)
    elif Type == 'Coiflet':
        if Par == 1:
            onFilter = torch.tensor([.038580777748, -.126969125396, -.077161555496,
                                .607491641386, .745687558934, .226584265197], dtype=dtype, device=device)
        elif Par == 2:
            onFilter = torch.tensor([.016387336463, -.041464936782, -.067372554722,
                                .386110066823, .812723635450, .417005184424,
                                -.076488599078, -.059434418646, .023680171947,
                                .005611434819, -.001823208871, -.000720549445], dtype=dtype, device=device)
        elif Par == 3:
            onFilter = torch.tensor([-.003793512864, .007782596426, .023452696142,
                                -.065771911281, -.061123390003, .405176902410,
                                .793777222626, .428483476378, -.071799821619,
                                -.082301927106, .034555027573, .015880544864,
                                -.009007976137, -.002574517688, .001117518771,
                                .000466216960, -.000070983303, -.000034599773], dtype=dtype, device=device)
        elif Par == 4:
            onFilter = torch.tensor([.000892313668, -.001629492013, -.007346166328,
                                .016068943964, .026682300156, -.081266699680,
                                -.056077313316, .415308407030, .782238930920,
                                .434386056491, -.066627474263, -.096220442034,
                                .039334427123, .025082261845, -.015211731527,
                                -.005658286686, .003751436157, .001266561929,
                                -.000589020757, -.000259974552, .000062339034,
                                .000031229876, -.000003259680, -.000001784985], dtype=dtype, device=device)
        elif Par == 5:
            onFilter = torch.tensor([-.000212080863, .000358589677, .002178236305,
                                -.004159358782, -.010131117538, .023408156762,
                                .028168029062, -.091920010549, -.052043163216,
                                .421566206729, .774289603740, .437991626228,
                                -.062035963906, -.105574208706, .041289208741,
                                .032683574283, -.019761779012, -.009164231153,
                                .006764185419, .002433373209, -.001662863769,
                                -.000638131296, .000302259520, .000140541149,
                                -.000041340484, -.000021315014, .000003734597,
                                .000002063806, -.000000167408, -.000000095158], dtype=dtype, device=device)
    elif Type == 'Daubechies':
        if Par == 4:
            onFilter = torch.tensor([.482962913145, .836516303738, .224143868042,
                                   -.129409522551], dtype=dtype, device=device)
        elif Par == 6:
            onFilter = torch.tensor([.332670552950, .806891509311, .459877502118,
                                 -.135011020010, -.085441273882, .035226291882], dtype=dtype, device=device)
        elif Par == 8:
            onFilter = torch.tensor([.230377813309, .714846570553, .630880767930,
                               -.027983769417, -.187034811719, .030841381836,
                                .032883011667, -.010597401785], dtype=dtype, device=device)
        elif Par == 10:
            onFilter = torch.tensor([.160102397974, .603829269797, .724308528438,
                                .138428145901, -.242294887066, -.032244869585,
                                .077571493840, -.006241490213, -.012580751999,
                                .003335725285], dtype=dtype, device=device)
        elif Par == 12:
            onFilter = torch.tensor([.111540743350, .494623890398, .751133908021,
                                .315250351709, -.226264693965, -.129766867567,
                                .097501605587, .027522865530, -.031582039317,
                                .000553842201, .004777257511, -.001077301085], dtype=dtype, device=device)
        elif Par == 14:
            onFilter = torch.tensor([.077852054085, .396539319482, .729132090846,
                                .469782287405, -.143906003929, -.224036184994,
                                .071309219267, .080612609151, -.038029936935,
                                -.016574541631, .012550998556, .000429577973,
                                -.001801640704, .000353713800], dtype=dtype, device=device)
        elif Par == 16:
            onFilter = torch.tensor([.054415842243, .312871590914, .675630736297,
                                .585354683654, -.015829105256, -.284015542962,
                                .000472484574, .128747426620, -.017369301002,
                                -.044088253931, .013981027917, .008746094047,
                                -.004870352993, -.000391740373, .000675449406,
                                -.000117476784], dtype=dtype, device=device)
        elif Par == 18:
            onFilter = torch.tensor([.038077947364, .243834674613, .604823123690,
                                 .657288078051, .133197385825, -.293273783279,
                                 -.096840783223, .148540749338, .030725681479,
                                 -.067632829061, .000250947115, .022361662124,
                                 -.004723204758, -.004281503682, .001847646883,
                                 .000230385764, -.000251963189, .000039347320], dtype=dtype, device=device)
        elif Par == 20:
            onFilter = torch.tensor([.026670057901, .188176800078, .527201188932,
                                .688459039454, .281172343661, -.249846424327,
                                -.195946274377, .127369340336, .093057364604,
                                -.071394147166, -.029457536822, .033212674059,
                                .003606553567, -.010733175483, .001395351747,
                                .001992405295, -.000685856695, -.000116466855,
                                .000093588670, -.000013264203], dtype=dtype, device=device)
    elif Type == 'Symmlet':
        if Par == 4:
            onFilter = torch.tensor([-.107148901418, -.041910965125, .703739068656,
                                1.136658243408, .421234534204, -.140317624179,
                                -.017824701442, .045570345896], dtype=dtype, device=device)
        elif Par == 5:
            onFilter = torch.tensor([.038654795955, .041746864422, -.055344186117,
                                .281990696854, 1.023052966894, .896581648380,
                                .023478923136, -.247951362613, -.029842499869,
                                .027632152958], dtype=dtype, device=device)
        elif Par == 6:
            onFilter = torch.tensor([.021784700327, .004936612372, -.166863215412,
                                -.068323121587, .694457972958, 1.113892783926,
                                .477904371333, -.102724969862, -.029783751299,
                                .063250562660, .002499922093, -.011031867509], dtype=dtype, device=device)
        elif Par == 7:
            onFilter = torch.tensor([.003792658534, -.001481225915, -.017870431651,
                                .043155452582, .096014767936, -.070078291222,
                                .024665659489, .758162601964, 1.085782709814,
                                .408183939725, -.198056706807, -.152463871896,
                                .005671342686, .014521394762], dtype=dtype, device=device)
        elif Par == 8:
            onFilter = torch.tensor([.002672793393, -.000428394300, -.021145686528,
                                .005386388754, .069490465911, -.038493521263,
                                -.073462508761, .515398670374, 1.099106630537,
                                .680745347190, -.086653615406, -.202648655286,
                                .010758611751, .044823623042, -.000766690896,
                                -.004783458512], dtype=dtype, device=device)
        elif Par == 9:
            onFilter = torch.tensor([.001512487309, -.000669141509, -.014515578553,
                                .012528896242, .087791251554, -.025786445930,
                                -.270893783503, .049882830959, .873048407349,
                                1.015259790832, .337658923602, -.077172161097,
                                .000825140929, .042744433602, -.016303351226,
                                -.018769396836, .000876502539, .001981193736], dtype=dtype, device=device)
        elif Par == 10:
            onFilter = torch.tensor([.001089170447, .000135245020, -.012220642630,
                                -.002072363923, .064950924579, .016418869426,
                                -.225558972234, -.100240215031, .667071338154,
                                1.088251530500, .542813011213, -.050256540092,
                                -.045240772218, .070703567550, .008152816799,
                                -.028786231926, -.001137535314, .006495728375,
                                .000080661204, -.000649589896], dtype=dtype, device=device)
    elif Type == 'Vaidyanathan':
        onFilter = torch.tensor([-.000062906118, .000343631905, -.000453956620,
                             -.000944897136, .002843834547, .000708137504,
                             -.008839103409, .003153847056, .019687215010,
                             -.014853448005, -.035470398607, .038742619293,
                             .055892523691, -.077709750902, -.083928884366,
                             .131971661417, .135084227129, -.194450471766,
                             -.263494802488, .201612161775, .635601059872,
                             .572797793211, .250184129505, .045799334111], dtype=dtype, device=device)
    elif Type == 'Battle':
        if Par == 1:
            onFilterTmp = torch.tensor([0.578163, 0.280931, -0.0488618, -0.0367309,
                                    0.012003, 0.00706442, -0.00274588,
                                    -0.00155701, 0.000652922, 0.000361781,
                                    -0.000158601, -0.0000867523], dtype=dtype, device=device)
        elif Par == 3:
            onFilterTmp = torch.tensor([0.541736, 0.30683, -0.035498, -0.0778079,
                                    0.0226846, 0.0297468, -0.0121455,
                                    -0.0127154, 0.00614143, 0.00579932,
                                    -0.00307863, -0.00274529, 0.00154624,
                                    0.00133086, -0.000780468, -0.00065562,
                                    0.000395946, 0.000326749, -0.000201818,
                                    -0.000164264, 0.000103307], dtype=dtype, device=device)
        elif Par == 5:
            onFilterTmp = torch.tensor([0.528374, 0.312869, -0.0261771, -0.0914068,
                                   0.0208414, 0.0433544, -0.0148537, -0.0229951,
                                0.00990635, 0.0128754, -0.00639886, -0.00746848,
                               0.00407882, 0.00444002, -0.00258816, -0.00268646,
                               0.00164132, 0.00164659, -0.00104207, -0.00101912,
                           0.000662836, 0.000635563, -0.000422485, -0.000398759,
                           0.000269842, 0.000251419, -0.000172685, -0.000159168,
                           0.000110709, 0.000101113], dtype=dtype, device=device)
        
        onFilter = torch.zeros(2*onFilterTmp.size(0)-1, dtype=dtype, device=device)
        onFilter[onFilterTmp.size(0)-1:2*onFilterTmp.size(0)] = onFilterTmp
        onFilter[0:onFilterTmp.size(0)-1] = torch.flip(onFilterTmp[1:], [0])
    
    if onFilter is None:
        raise ValueError(f"Unknown filter type: {Type}")
    
    return onFilter / torch.linalg.norm(onFilter)


def MirrorFilt(x: torch.Tensor) -> torch.Tensor:
    """
    Apply (-1)^t modulation (PyTorch version, optimized).
    
    This is a translation of the original Matlab implementation of
    MirrorFilt.m from the WaveLab850 toolbox.
    
    Optimized: Uses 1 - 2*(n%2) instead of pow(-1, n) to avoid temporary tensors.

    Args:
        x: 1D input tensor

    Returns:
        Modulated signal with DC frequency content shifted to Nyquist frequency
    """
    n = x.size(0)
    # (-1)^n = 1 - 2*(n%2) is faster than torch.pow
    indices = torch.arange(n, dtype=torch.int64, device=x.device)
    modulation = (1 - 2 * (indices % 2)).to(x.dtype)
    return modulation * x


def modulate2(x: torch.Tensor, type: str, 
              center: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    2D modulation (PyTorch version, optimized).
    
    This is a translation of the original Matlab implementation of
    modulate2.m from the Nonsubsampled Contourlet Toolbox.
    
    Optimized: Uses integer arithmetic for modulation pattern.

    Args:
        x: 1D or 2D input tensor
        type: 'r', 'c' or 'b' for modulate along row, column or both
        center: Origin of modulation as floor(size(x)/2)+1+center (default: [0, 0])

    Returns:
        Modulated tensor
    """
    device = x.device
    dtype = x.dtype
    
    if x.dim() == 1:
        sz0 = x.size(0)
        if center is None:
            origin = sz0 // 2 + 1
        else:
            c = int(center.item()) if isinstance(center, torch.Tensor) else int(center)
            origin = sz0 // 2 + 1 + c
        
        # Compute modulation: (-1)^(i - origin + 1) for i in 0..sz0-1
        indices = torch.arange(sz0, dtype=torch.int64, device=device)
        shifts = indices - origin + 1
        m = (1 - 2 * (shifts.abs() % 2)).to(dtype)
        return x * m
    else:
        sz0, sz1 = x.shape
        if center is None:
            origin0 = sz0 // 2 + 1
            origin1 = sz1 // 2 + 1
        else:
            origin0 = sz0 // 2 + 1 + int(center[0].item() if isinstance(center, torch.Tensor) else center[0])
            origin1 = sz1 // 2 + 1 + int(center[1].item() if isinstance(center, torch.Tensor) else center[1])
        
        # Compute modulation indices
        n1 = torch.arange(sz0, dtype=torch.int64, device=device) - origin0 + 1
        n2 = torch.arange(sz1, dtype=torch.int64, device=device) - origin1 + 1
        
        if type == 'r':
            # Modulate along rows
            m1 = (1 - 2 * (n1.abs() % 2)).to(dtype)
            return x * m1.unsqueeze(1)
        elif type == 'c':
            # Modulate along columns
            m2 = (1 - 2 * (n2.abs() % 2)).to(dtype)
            return x * m2.unsqueeze(0)
        elif type == 'b':
            # Modulate along both
            m1 = (1 - 2 * (n1.abs() % 2)).to(dtype)
            m2 = (1 - 2 * (n2.abs() % 2)).to(dtype)
            m = torch.outer(m1, m2)
            return x * m
        else:
            raise ValueError(f"Unknown modulation type: {type}")


def dmaxflat(N: int, d: int, 
             dtype: torch.dtype = torch.float64, 
             device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Generate 2D diamond maxflat filters of order N (PyTorch version).
    
    This is a translation of the original Matlab implementation of dmaxflat.m
    from the Nonsubsampled Contourlet Toolbox.

    Args:
        N: Order of the filter, must be in {1, 2, ..., 7}
        d: Value for the (0,0) coefficient, being 1 or 0
        dtype: Output tensor dtype (default: torch.float64)
        device: Device for the output tensor (default: 'cpu')

    Returns:
        2D diamond maxflat filter tensor
    """
    if (N > 7) or (N < 1):
        raise ValueError('N must be in {1, 2, ..., 7}')
    
    if N == 1:
        h = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype, device=device) / 4
        h[1, 1] = d
    elif N == 2:
        h = torch.tensor([[0, -1, 0], [-1, 0, 10], [0, 10, 0]], dtype=dtype, device=device)
        h = torch.cat([h, torch.flip(h[:, :-1], [1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], [0])], dim=0) / 32
        h[2, 2] = d
    elif N == 3:
        h = torch.tensor([[0, 3, 0, 2], [3, 0, -27, 0], [0, -27, 0, 174],
                          [2, 0, 174, 0]], dtype=dtype, device=device)
        h = torch.cat([h, torch.flip(h[:, :-1], [1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], [0])], dim=0)
        h[3, 3] = d
    elif N == 4:
        h = torch.tensor([[0, -5, 0, -3, 0], [-5, 0, 52, 0, 34],
                          [0, 52, 0, -276, 0], [-3, 0, -276, 0, 1454],
                          [0, 34, 0, 1454, 0]], dtype=dtype, device=device) / (2**12)
        h = torch.cat([h, torch.flip(h[:, :-1], [1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], [0])], dim=0)
        h[4, 4] = d
    elif N == 5:
        h = torch.tensor([[0, 35, 0, 20, 0, 18], [35, 0, -425, 0, -250, 0],
                    [0, -425, 0, 2500, 0, 1610], [20, 0, 2500, 0, -10200, 0],
                    [0, -250, 0, -10200, 0, 47780],
                    [18, 0, 1610, 0, 47780, 0]], dtype=dtype, device=device) / (2**17)
        h = torch.cat([h, torch.flip(h[:, :-1], [1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], [0])], dim=0)
        h[5, 5] = d
    elif N == 6:
        h = torch.tensor([[0, -63, 0, -35, 0, -30, 0],
                     [-63, 0, 882, 0, 495, 0, 444],
                     [0, 882, 0, -5910, 0, -3420, 0],
                     [-35, 0, -5910, 0, 25875, 0, 16460],
                     [0, 495, 0, 25875, 0, -89730, 0],
                     [-30, 0, -3420, 0, -89730, 0, 389112],
                     [0, 44, 0, 16460, 0, 389112, 0]], dtype=dtype, device=device) / (2**20)
        h = torch.cat([h, torch.flip(h[:, :-1], [1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], [0])], dim=0)
        h[6, 6] = d
    elif N == 7:
        h = torch.tensor([[0, 231, 0, 126, 0, 105, 0, 100],
                    [231, 0, -3675, 0, -2009, 0, -1715, 0],
                    [0, -3675, 0, 27930, 0, 15435, 0, 13804],
                    [126, 0, 27930, 0, -136514, 0, -77910, 0],
                    [0, -2009, 0, -136514, 0, 495145, 0, 311780],
                    [105, 0, 15435, 0, 495145, 0, -1535709, 0],
                    [0, -1715, 0, -77910, 0, -1534709, 0, 6305740],
                    [100, 0, 13804, 0, 311780, 0, 6305740, 0]], dtype=dtype, device=device) / (2**24)
        h = torch.cat([h, torch.flip(h[:, :-1], [1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], [0])], dim=0)
        h[7, 7] = d
    
    return h


def _convolve2d_full(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with 'full' mode (like scipy.signal.convolve2d mode='full').
    
    Args:
        input: 2D input tensor
        kernel: 2D convolution kernel

    Returns:
        Convolved output with size = input.shape + kernel.shape - 1
    """
    # Flip kernel for convolution (not correlation)
    kernel_flipped = torch.flip(kernel, [0, 1])
    
    # Compute padding for 'full' mode
    pad_h = kernel.shape[0] - 1
    pad_w = kernel.shape[1] - 1
    
    # Add batch and channel dimensions
    input_4d = input.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel_flipped.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution with padding
    output = F.conv2d(input_4d, kernel_4d, padding=(pad_h, pad_w))
    
    return output.squeeze(0).squeeze(0)


def mctrans(b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    McClellan transformation (PyTorch version).
    
    This is a translation of the original Matlab implementation of mctrans.m
    from the Nonsubsampled Contourlet Toolbox by Arthur L. da Cunha.
    
    Produces the 2-D FIR filter H that corresponds to the 1-D FIR filter b
    using the transform t.

    Args:
        b: 1D input filter tensor
        t: 2D transformation matrix tensor

    Returns:
        2D transformed filter tensor
    """
    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    n = (b.size(0) - 1) // 2
    
    # Replicate NumPy: b = fftlib.fftshift(b[::-1]); b = b[::-1]
    b_shifted = torch.fft.fftshift(b.flip([0]))
    b_shifted = b_shifted.flip([0])
    
    a = torch.zeros(n + 1, dtype=b.dtype, device=b.device)
    a[0] = b_shifted[0]
    a[1:n+1] = 2 * b_shifted[1:n+1]
    
    inset = torch.floor((torch.tensor(t.shape, dtype=torch.float64, device=b.device) - 1) / 2).to(torch.int64)
    inset_0 = int(inset[0].item())
    inset_1 = int(inset[1].item())
    
    # Use Chebyshev polynomials to compute h
    P0 = 1  # Start as Python scalar like NumPy
    P1 = t.clone()
    h = a[1] * P1
    
    # rows and cols start as 1-element tensors to maintain type consistency
    rows = torch.tensor([inset_0 + 1], device=b.device, dtype=torch.int64)
    cols = torch.tensor([inset_1 + 1], device=b.device, dtype=torch.int64)
    h[int(rows[0].item())-1, int(cols[0].item())-1] = h[int(rows[0].item())-1, int(cols[0].item())-1] + a[0] * P0
    
    for i in range(3, n + 2):
        P2 = 2 * _convolve2d_full(t, P1)
        
        # NumPy: rows = (rows + inset[0]).astype(int)
        # This increments rows at loop START
        rows = rows + inset_0
        cols = cols + inset_1
        
        if rows.numel() == 1:
            # First iteration (i=3): rows is still 1-element tensor
            r = int(rows[0].item())
            c = int(cols[0].item())
            P2[r-1, c-1] = P2[r-1, c-1] - P0
        else:
            # rows is a tensor/array - add inset to each element
            r_start = int(rows[0].item()) - 1
            r_end = int(rows[-1].item())
            c_start = int(cols[0].item()) - 1
            c_end = int(cols[-1].item())
            P2[r_start:r_end, c_start:c_end] = P2[r_start:r_end, c_start:c_end] - P0
        
        # NumPy: rows = inset[0] + np.arange(P1.shape[0]) + 1
        # Now rows becomes an array
        rows = inset[0] + torch.arange(P1.shape[0], device=b.device, dtype=torch.int64) + 1
        cols = inset[1] + torch.arange(P1.shape[1], device=b.device, dtype=torch.int64) + 1
        
        hh = h.clone()
        h = a[i-1] * P2
        
        r_start = int(rows[0].item()) - 1
        r_end = int(rows[-1].item())
        c_start = int(cols[0].item()) - 1
        c_end = int(cols[-1].item())
        h[r_start:r_end, c_start:c_end] = h[r_start:r_end, c_start:c_end] + hh
        
        P0 = P1.clone()  # P0 becomes a matrix
        P1 = P2.clone()
    
    h = torch.rot90(h, 2)
    return h


def dfilters(fname: str, type: str,
             dtype: torch.dtype = torch.float64,
             device: Union[str, torch.device] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate directional 2D filters (PyTorch version).
    
    This is a translation of the original Matlab implementation of dfilters.m
    from the Nonsubsampled Contourlet Toolbox.

    Args:
        fname: Filter name. Available options:
               'haar', 'vk', 'ko', 'kos', 'lax', 'sk', 'cd', 'dvmlp',
               'oqf_362', 'dmaxflat4', 'dmaxflat5', 'dmaxflat6', 'dmaxflat7'
        type: 'd' or 'r' for decomposition or reconstruction filters
        dtype: Output tensor dtype (default: torch.float64)
        device: Device for the output tensor (default: 'cpu')

    Returns:
        Tuple of (h0, h1) diamond filter pair (lowpass and highpass)
    """
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    
    if fname == 'haar':
        if type.lower() == 'd':
            h0 = torch.tensor([1, 1], dtype=dtype, device=device) / sqrt2
            h1 = torch.tensor([-1, 1], dtype=dtype, device=device) / sqrt2
        else:
            h0 = torch.tensor([1, 1], dtype=dtype, device=device) / sqrt2
            h1 = torch.tensor([1, -1], dtype=dtype, device=device) / sqrt2
    elif fname == 'vk':
        if type.lower() == 'd':
            h0 = torch.tensor([1, 2, 1], dtype=dtype, device=device) / 4
            h1 = torch.tensor([-1, -2, 6, -2, -1], dtype=dtype, device=device) / 4
        else:
            h0 = torch.tensor([-1, 2, 6, 2, -1], dtype=dtype, device=device) / 4
            h1 = torch.tensor([-1, 2, -1], dtype=dtype, device=device) / 4
        t = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype, device=device) / 4
        h0 = mctrans(h0, t)
        h1 = mctrans(h1, t)
    elif fname == 'ko':
        a0 = 2.0
        a1 = 0.5
        a2 = 1.0
        h0 = torch.tensor([[0,    -a1,  -a0*a1, 0],
                        [-a2, -a0*a2, -a0,  1],
                        [0, a0*a1*a2, -a1*a2, 0]], dtype=dtype, device=device)
        h1 = torch.tensor([[0, -a1*a2, -a0*a1*a2, 0],
                       [1,   a0,  -a0*a2,  a2],
                       [0, -a0*a1,   a1,   0]], dtype=dtype, device=device)
        norm = sqrt2 / h0.sum()
        h0 = h0 * norm
        h1 = h1 * norm
        if type == 'r':
            h0 = torch.flip(h0, [0])
            h1 = torch.flip(h1, [0])
    elif fname == 'kos':
        sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=dtype, device=device))
        a0 = -sqrt3
        a1 = -sqrt3
        a2 = 2 + sqrt3
        h0 = torch.tensor([[0,    -a1,  -a0*a1, 0],
                        [-a2, -a0*a2, -a0,  1],
                        [0, a0*a1*a2, -a1*a2, 0]], dtype=dtype, device=device)
        h1 = torch.tensor([[0, -a1*a2, -a0*a1*a2, 0],
                       [1,   a0,  -a0*a2,  a2],
                       [0, -a0*a1,   a1,   0]], dtype=dtype, device=device)
        norm = sqrt2 / h0.sum()
        h0 = h0 * norm
        h1 = h1 * norm
        if type == 'r':
            h0 = torch.flip(h0, [0])
            h1 = torch.flip(h1, [0])
    elif fname == 'cd' or fname == '7-9':
        h0_1d = torch.tensor([0.026748757411, -0.016864118443, -0.078223266529,
                    0.266864118443, 0.602949018236, 0.266864118443,
                    -0.078223266529, -0.016864118443, 0.026748757411], dtype=dtype, device=device)
        g0_1d = torch.tensor([-0.045635881557, -0.028771763114, 0.295635881557,
                       0.557543526229, 0.295635881557, -0.028771763114,
                        -0.045635881557], dtype=dtype, device=device)
        if type == 'd':
            h1_1d = modulate2(g0_1d, 'c')
        else:
            h1_1d = modulate2(h0_1d, 'c')
            h0_1d = g0_1d
        t = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype, device=device) / 4
        h0 = sqrt2 * mctrans(h0_1d, t)
        h1 = sqrt2 * mctrans(h1_1d, t)
    elif fname == 'oqf_362':
        sqrt15 = torch.sqrt(torch.tensor(15.0, dtype=dtype, device=device))
        sqrt2_val = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        h0 = sqrt2 / 64 * torch.tensor([[sqrt15, -3, 0],
                        [0, 5, sqrt15], [-2*sqrt2_val, 30, 0],
                        [0, 30, 2*sqrt15], [sqrt15, 5, 0],
                        [0, -3, -sqrt15]], dtype=dtype, device=device)
        h1 = -modulate2(h0, 'b')
        h1 = -torch.flip(h1, [0])
        if type == 'r':
            h0 = torch.flip(h0, [0])
            h1 = -modulate2(h0, 'b')
            h1 = -torch.flip(h1, [0])
    elif fname == 'dmaxflat4':
        M1 = 1/sqrt2
        M2 = M1.clone()
        k1 = 1 - sqrt2
        k3 = k1.clone()
        k2 = M1.clone()
        h_1d = torch.tensor([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3], dtype=dtype, device=device) * M1
        h_1d = torch.cat([h_1d, torch.flip(h_1d[:-1], [0])])
        g_1d = torch.tensor([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2], dtype=dtype, device=device) * M2
        g_1d = torch.cat([g_1d, torch.flip(g_1d[:-1], [0])])
        B = dmaxflat(4, 0, dtype=dtype, device=device)
        h0 = mctrans(h_1d, B)
        g0 = mctrans(g_1d, B)
        h0 = sqrt2 * h0 / h0.sum()
        g0 = sqrt2 * g0 / g0.sum()
        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    elif fname == 'dmaxflat5':
        M1 = 1/sqrt2
        M2 = M1.clone()
        k1 = 1 - sqrt2
        k3 = k1.clone()
        k2 = M1.clone()
        h_1d = torch.tensor([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3], dtype=dtype, device=device) * M1
        h_1d = torch.cat([h_1d, torch.flip(h_1d[:-1], [0])])
        g_1d = torch.tensor([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2], dtype=dtype, device=device) * M2
        g_1d = torch.cat([g_1d, torch.flip(g_1d[:-1], [0])])
        B = dmaxflat(5, 0, dtype=dtype, device=device)
        h0 = mctrans(h_1d, B)
        g0 = mctrans(g_1d, B)
        h0 = sqrt2 * h0 / h0.sum()
        g0 = sqrt2 * g0 / g0.sum()
        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    elif fname == 'dmaxflat6':
        M1 = 1/sqrt2
        M2 = M1.clone()
        k1 = 1 - sqrt2
        k3 = k1.clone()
        k2 = M1.clone()
        h_1d = torch.tensor([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3], dtype=dtype, device=device) * M1
        h_1d = torch.cat([h_1d, torch.flip(h_1d[:-1], [0])])
        g_1d = torch.tensor([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2], dtype=dtype, device=device) * M2
        g_1d = torch.cat([g_1d, torch.flip(h_1d[:-1], [0])])
        B = dmaxflat(6, 0, dtype=dtype, device=device)
        h0 = mctrans(h_1d, B)
        g0 = mctrans(g_1d, B)
        h0 = sqrt2 * h0 / h0.sum()
        g0 = sqrt2 * g0 / g0.sum()
        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    elif fname == 'dmaxflat7':
        M1 = 1/sqrt2
        M2 = M1.clone()
        k1 = 1 - sqrt2
        k3 = k1.clone()
        k2 = M1.clone()
        h_1d = torch.tensor([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3], dtype=dtype, device=device) * M1
        h_1d = torch.cat([h_1d, torch.flip(h_1d[:-1], [0])])
        g_1d = torch.tensor([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2], dtype=dtype, device=device) * M2
        g_1d = torch.cat([g_1d, torch.flip(h_1d[:-1], [0])])
        B = dmaxflat(7, 0, dtype=dtype, device=device)
        h0 = mctrans(h_1d, B)
        g0 = mctrans(g_1d, B)
        h0 = sqrt2 * h0 / h0.sum()
        g0 = sqrt2 * g0 / g0.sum()
        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    else:
        raise ValueError(f"Unknown filter name: {fname}")
    
    return h0, h1
