from spaudiopy import sph
import numpy as np
import warnings
import scipy.special
import scipy.constants
import librosa
import matplotlib.pyplot as plt

def a2b(N, audio_numpy, capsule_coords):

    """
    encodes recordings from microphone array
    capsules (raw A-format) to B-format
    ambisonics using the pseudo_inverse of
    the spherical harmonics matrix

    Args:
        N (int): the order of B format
        audio_numpy (np.array): the numpy array with the audio
        capsule_coords (dict): dictionary with channel names and
            corresponding coordinates in polar form (colatitude
            and azimuth in radians)

    Returns:
        a numpy array with the encoded B format
    """

    coords_numpy = np.array([c for c in capsule_coords.values()])
    SH = sph.sh_matrix(N, coords_numpy[:, 1], coords_numpy[:, 0], "real")
    Y = np.linalg.pinv(SH)
    return np.dot(Y, audio_numpy)

def array2sh_process(
    N, audio_numpy, capsule_coords, fs
):
    """
    Encodes ambisonics B-format signal
    from raw recorded microphone signals
    using frequency-dependent, equalized
    spherical harmonic weights. 
    
    This is a translation of the eigenmike 
    encoding pipeline from the SPARTA
    array2sh plugin by Leo McCormack.

    Args:
        N (int): the order of B format
        audio_numpy (np.array): the numpy array with the audio
        capsule_coords (dict): dictionary with channel names and
            corresponding coordinates in polar form (colatitude
            and azimuth in radians)
        fs (int): sample rate

    Returns:
        a numpy array with the encoded B format
    """
        
    nSH = (N+1)**2
    Q = len(audio_numpy)
    r = 0.042
    R = 0.042
    #Below STFT doesn't suffice as a time-frequency domain representation for ambisonics encoding.
    #Further work is needed to achieve the same time-frequency representation as Array2SH
    audio_stft = librosa.stft(audio_numpy,n_fft=264,hop_length=128,win_length=128,window='boxcar')#Shape = [32,133,25688]
    nBands = audio_stft.shape[1]
    coords_numpy = np.array([c for c in capsule_coords.values()])
    W = calculate_sht_matrix(r,R,nBands,N,nSH,fs,coords_numpy,Q)
    
    sh_spectrum = np.zeros((nSH,133,audio_stft.shape[2]),dtype='complex128')
    for i in range(audio_stft.shape[-1]):
        spectrum = audio_stft[...,i]
        for currBand in range(audio_stft.shape[1]):
            sh_spectrum[:,currBand,i] = np.dot(spectrum[...,currBand],W[currBand,:,:].T).T

    audio_istft = librosa.istft(sh_spectrum,hop_length=128,win_length=128,window='boxcar')
    return audio_istft

def calculate_sht_matrix(
    r,R,HYBRID_BANDS,order,nSH,fs,coords_numpy,Q
):
    """
    calculates the matrix by which the time-frequency-domain 
    sensor signals will be multiplied with to get spherical 
    harmonic signals in the frequency domain.

    See section 2 figure 3:
    https://leomccormack.github.io/sparta-site/docs/help/related-publications/mccormack2018real.pdf

    Args:
        r (float): Radius of sensors from center of array in meters
        R (float): Radius of baffle from center of array in meters (if baffle exists)
        HYBRID_BANDS (int): nfft
        order (int): desired order of ambisonics encoding
        nSH (int): desired number of ambisonics channels
        fs (int): sample rate
        coords_numpy (np.array): coordinates of the microphone array
        Q (int): number of array sensors

    Returns:
        np.array: 3D array containing encoding matrix (HYBRID_BANDS x nSH x Q)
    """

    #-------------------- VARIABLE CREATION --------------------
    kr = np.zeros(HYBRID_BANDS)
    #kR = np.zeros(HYBRID_BANDS)   #for baffle radius =/= sensor radius
    fv = np.fft.fftfreq((HYBRID_BANDS-1)*2,1.0/fs)
    freqVector = fv[:HYBRID_BANDS]
    freqVector[-1] = freqVector[-1]*(-1)
    c = scipy.constants.speed_of_sound #speed of sound in air
    for band in range(HYBRID_BANDS):
        kr[band] = 2*np.pi*freqVector[band]*r/c
        #kR[band] = 2*np.pi*freqVector[band]*R/c

    #---------------------- CALCULATE Y_e ---------------------- ("Spherical harmonic weights for each sensor direction")
    SH = sph.sh_matrix(order, coords_numpy[:, 1], coords_numpy[:, 0], "real")
    #Y_mic = (SH.T/(np.sqrt(1/(4*np.pi)))) 
    pinv_Y_mic = np.linalg.pinv(SH/(np.sqrt(1/(4*np.pi))))
    pinv_Y_mic_cmplx = pinv_Y_mic + 0j

    #----------- SWITCH CASES FOR ARRAY CONSTRUCTION -----------
    #if: filter type is soft limiter or Tikhonov
    #BEGIN case ARRAY_SPHERICAL
    #BEGIN case WEIGHT_RIGID_DIPOLE
    if(r == R):
        bN = np.array(calculate_modal_coefficients(order, kr, HYBRID_BANDS, None, None, freqVector))
    #else: #Case not for Eigenmike
    #END case WEIGHT_RIGID_DIPOLE
    #END case ARRAY_SPHERICAL
       
    #------ CREATE W_l FROM THEORETICAL MODAL COEFFICENTS ------
    bN_norm = bN/(4*np.pi)
    bN_reg_inv = 1/(bN_norm+1e-24)#"direct inverse" method, see paper
    regPar = 15 #(15 was printed in eigenmike encoding in SAF)

    #---------------- REGULARIZED INVERSE METHODS ----------------  
    #if: filterType == FILTER_TIKHONOV (regularization method used in SAF test)
    alpha = (Q**.5) * (10**regPar/20)
    bN_inv = np.zeros([HYBRID_BANDS, order+1], dtype = 'complex128')#dimensions in EM conversion:133x5
    for band in range(HYBRID_BANDS):
        for n in range(order+1):
            beta = ((1-(1.0-1.0 / (alpha**2)**.5))/(1 + (1.0-1.0 / (alpha**2))**.5))**.5
            bN_inv[band][n] = np.conj(bN_reg_inv[band*(order+1)+n]) / np.abs(bN_reg_inv[band*(order+1)+n])**2 + beta**2 + 0j
    #endif: filterType == FILTER_TIKHONOV

    #---------- COMPUTE ENCODING MATRIX: W = W_l x Y_e ---------- 
    bN_inv_R = array2sh_replicate_order(bN_inv, order, HYBRID_BANDS)
    diag_bN_inv_R = np.zeros((nSH,nSH), dtype=np.complex128)
    W = np.zeros((HYBRID_BANDS, nSH, Q), dtype = 'complex128') #nSH was MAX_NUM_SH_SIGNALS, Q was MAX_NUM_SENSORS
    for band in range(HYBRID_BANDS):
        for i in range(nSH):
            diag_bN_inv_R[i][i] = bN_inv_R[band][i] 
        W[band,:,:] = np.dot(diag_bN_inv_R,pinv_Y_mic_cmplx)
        
    return W

def array2sh_replicate_order(
    bN_inv, order, HYBRID_BANDS
):
    '''
    "Takes the bNs computed up to N+1, and replicates them to be of length
    (N+1)^2 (replicating the 1st order bNs 3 times, 2nd -> 5 times etc.)"

    Args:
        bN_inv (np.array): Inverse modal coefficients
        order (int): desired encoded order
        HYBRID_BANDS (int): nfft

    Returns:
        np.array: 2d array with inverse bN's in desired arrangement
    '''

    MAX_SH_ORDER = 7 #This is defined in _common.h
    bN_inv_R = np.zeros((HYBRID_BANDS,(MAX_SH_ORDER+1)**2), dtype = 'complex128')#size determined at line 117 in array2sh_internal.h
    o= np.zeros(MAX_SH_ORDER + 2 , dtype='int64')
    o[0:order+2] = np.arange(order+2)
    o=o*o
    for band in range(HYBRID_BANDS):
        for n in range(order+1):
            for i in range(o[n], o[n+1]):
                bN_inv_R[band][i] = bN_inv[band][n]
        
    return bN_inv_R

def calculate_modal_coefficients(
    order, kr, nBands, arrayType, dirCoeff,fv
): 
    """
    computes "theoretical modal coefficients" for ambisonics
    encoding. These coefficients "take into account whether 
    the array construction is open or rigid and the directivity 
    of the sensors (cardioid, dipole, or omnidirectional)"

    See section 2 figure 7:
    https://leomccormack.github.io/sparta-site/docs/help/related-publications/mccormack2018real.pdf

    Args:
        order (int):
        kr (np.array):
        nBands (int):
        arrayType (string):
        dirCoeff (int): (Not used in Eigenmike encoding)

    Returns:
        np.array: theoretical modal coefficients
    """
    
    #Variable creation before switch case in SAF
    b_N = np.zeros((nBands*(order+1)),dtype=np.complex_)

    #Below is only case ARRAY_CONSTRUCTION_RIGID in saf_sh.c
    jn = bessel_functions(order, kr, False)
    jnprime = bessel_functions(order, kr, True)
    hn2 = hankel2_functions(order, kr, False)
    hn2prime = hankel2_functions(order, kr, True)
    maxN=1000000000    
    #Below: SAF bessel functions return a maxN_tmp
    #if maxN_tmp<maxN: maxN = maxN_tmp
    #if maxN_tmp<maxN: maxN = maxN_tmp  #maxN is "the minimum highest order that was computed for all values in kr"
    
    maxN = order #This is my solution to the above if-statements being commented out
                #saf_example_array2sh which converts an eigenmike file sets maxN to 4 at this point
    
    # "modal coefficients for rigid spherical array: 4*pi*1i^n * (jn-(jnprime./hn2prime).*hn2)"
    for i in range(0,nBands):
        for n in range(0,maxN+1):
            if(n==0 and kr[i]<=1e-20):
                b_N[i*(order+1)+n]= 4*np.pi + 0j 
            elif(kr[i] <= 1e-20):
                b_N[i*(order+1)+n] = 0+0*1j
            else:
                b_N[i*(order+1)+n] = (np.power((0+1j),n) * 4 * np.pi)*(jn[i*(order+1)+n] - ((jnprime[i*(order+1)+n] / hn2prime[i*(order+1)+n]) * hn2[i*(order+1)+n])) 

    return b_N

def hankel2_functions(
    order, kr, is_derivatives
):
    """
    computes array of spherical hankel functions of the
    second kind from orders 0 to 'order' for domain kr 

    Args:
        order (int): the order of B format
        kr (np.array): the numpy array 
        is_derivatives (boolean): determines if function 
            returns derivatives of bessel values

    Returns:
        np.array: Desired hankel values
    """

    arr = [0.j]*(order+1)
    for i in range(order+1):
        arr[i] = (scipy.special.spherical_jn(i, kr, is_derivatives) - (scipy.special.spherical_yn(i, kr, is_derivatives)*1j)).tolist()
    return np.array(arr).flatten('F')

def bessel_functions(
    order, kr, is_derivatives
):
    """
    computes array of spherical bessel functions of the
    first kind from orders 0 to 'order' for domain kr 

    Args:
        order (int): the order of B format
        kr (np.array): the numpy array 
        is_derivatives (boolean): determines if function 
            returns derivatives of bessel values

    Returns:
        np.array: Desired bessel values
    """
    
    arr = [0.]*(order+1)
    for i in range(order+1):
        arr[i] = scipy.special.spherical_jn(i, kr, is_derivatives).tolist()
    return np.array(arr).flatten('F')

def _get_audio_numpy(
    clip_names, dataset, fmt_in, fmt_out, capsule_coords=None, N=None, fs=None
):

    """
    combine clips that correspond to a multitrack recording
    into a numpy array and return it in A or B format.

    Args:
        clip_names (list): list of strings with names of clips
            to be loaded (and combined if different clips have
            the recording by different microphone capsules).
        dataset (soundata.Dataset): the soundata dataset where
            the clips can be loaded from
        fmt_in (str): whether the clips originally are in A or B
            format
        fmt_out (str): the target format (A or B). Currently it only
            works A->B
        capsule_coords (dict): dictionary with channel names and
            corresponding coordinates in polar form (colatitude
            and azimuth in radians)
        N (int): the order of B format
        fs (int): the target sampling rate

    Returns:
        audio_array (np.array): the numpy array with the audio
    """
    all_dataset_clip_names = dataset.load_clips()
    if fmt_in not in ["A", "B"] or fmt_out not in ["A", "B"]:
        raise ValueError(
            "the input and output formats should be either 'A' or 'B' but fmt_in is {} and fmt_out is {}".format(
                fmt_in, fmt_out
            )
        )
    if fmt_in == "B" and fmt_out == "A":
        raise ValueError("B to A conversion currently not supported")
    if fmt_in == "A" and fmt_out == "B" and capsule_coords == None:
        raise ValueError(
            "To convert between A and B format you must specify capsule coordinates"
        )
    audio_data = [all_dataset_clip_names[ac].audio for ac in clip_names]
    audio_array, audio_fs = list(map(list, zip(*audio_data)))
    audio_array = np.squeeze(np.array(audio_array))
    if fmt_out == "B" and N != None and (N + 1) ** 2 > len(audio_array):
        raise ValueError(
            "(N+1)^2 should be less than or equal to the number of capsules being converted to B format, but (N+1)^2 is {} and the number of capsules is {}".format(
                (N + 1) ** 2, len(audio_array)
            )
        )
    audio_fs = audio_fs[0]
    if fs != None and audio_fs != fs:
        audio_array = librosa.resample(audio_array, audio_fs, fs)
    if fmt_in == fmt_out:
        if N != None:
            warnings.warn(UserWarning("N parameter was specified but not used"))
        return audio_array
    if fmt_in == "A" and fmt_out == "B":
        N = int(np.sqrt(len(clip_names)) - 1) if N == None else N
        #audio_array = a2b(N, audio_array, capsule_coords)
        audio_array = array2sh_process(N, audio_array, capsule_coords,fs)
        return audio_array
