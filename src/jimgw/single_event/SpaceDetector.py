#!/usr/bin/env python

"""
LISA code written by Karel Plets - graphing the SNR spectrum
Date: 28/02/2025

Notes:  ---------- CUMULATIVE VERSION-----------------
        
"""

############################################
#############  IMPORT PACKAGES #############
############# DEFINE VARIABLES #############
############################################

# LISA RELEVANT PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import h5py
from fastlisaresponse import pyResponseTDI, ResponseWrapper
from astropy import units as un

from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
YRSID_SI = 31558149.763545603

equal = EqualArmlengthOrbits()
equal.configure(linear_interp_setup=True)


import time

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.single_event.detector import Detector
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
from jimgw.single_event.utils import Mc_q_to_m1_m2
from flowMC.strategy.optimization import optimization_Adam



class SpaceBasedDetector(Detector):
    polarization_mode: list[Polarization]
    frequencies: Float[Array, " n_sample"]
    data: Float[Array, " n_sample 3"]
    psd: Float[Array, " n_sample 3 3"]

    # giving the values for one of the three detectors
    # the rest will be automatically calculated
    '''
    latitude: Float = 0
    longitude: Float = 0
    xarm_azimuth: Float = 0
    yarm_azimuth: Float = 0
    xarm_tilt: Float = 0
    yarm_tilt: Float = 0
    elevation: Float = 0'''
    # I don't think these will be as relevant
    
    
    # New variables I want to add
    orbit: str = 'equal'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __init__(self, name: str, **kwargs) -> None: # To be changed into stuff you need for FD analysis
        self.name = name

        '''
        I think these are useless parameters for LISA?
        
        self.latitude = kwargs.get("latitude", 0)
        self.longitude = kwargs.get("longitude", 0)
        self.elevation = kwargs.get("elevation", 0)
        self.xarm_azimuth = kwargs.get("xarm_azimuth", 0)
        self.yarm_azimuth = kwargs.get("yarm_azimuth", 0)
        self.xarm_tilt = kwargs.get("xarm_tilt", 0)
        self.yarm_tilt = kwargs.get("yarm_tilt", 0)
        modes = kwargs.get("mode", "pc")
        '''

        self.polarization_mode = [Polarization(m) for m in modes]
        self.frequencies = jnp.array([])
        self.data = jnp.array([])
        self.psd = jnp.array([])
        
        # Newly added variables by Karel
        self.orbit = kwargs.get('orbit','equal') # equal or ESA orbit. for response function of fastlisaresponse
        self.channel = kwargs.get('tdi channel', 'XYZ') # AET, XYZ or perhaps later Sagnac
        self.tdi_gen = kwargs.get('tdi_gen', '1st generation')
        self.use_gpu = kwargs.get('use_gpu', True ) 
        
        self.order = kwargs.get('order',25) # order of lagrangian interpolation, fastlisaresponse parameter
        self.index_lambda = kwargs.get('index_lambda',6) # fastlisaresponse parameter, i forgot what exactly
        self.index_beta = kwargs.get('index_beta',7) # fastlisaresponse parameter, i forgot what exactly
    
    @staticmethod
    def get_orbit(self):
        if self.orbit == "equal":
            orbit = EqualArmlengthOrbits
            orbit.configure(linear_interp_setup=True)
            return orbit
        elif self.orbit == "ESA":
            orbit = ESAOrbits
            orbit.configure(linear_interp_setup=True)
            return orbit
        else:
            raise NotImplementedError

    @staticmethod
    def _get_arm(
        self,
        t: Float,
    ) -> Float[Array, " 3"]: # To be changed into stuff you can find in lisaanalysistools
        """
        Construct detector-arm vectors in Barycentric Solar system's Cartesian coordinates.

        Parameters
        ---------
        t    : point of time in the orbit
        orbit: orbit type (ESAOrbit or EqualArmsOrbit), taken from lisatools
        """
        raise NotImplementedError
        
    @property
    def arms(self) -> tuple[Float[Array, " 3"],
                            Float[Array, " 3"],
                            Float[Array, " 3"]
                            ]:
        """
        Detector arm vectors (x, y, z).

        Returns
        -------
        x : Float[Array, " 3"]
            x-arm vector.
        y : Float[Array, " 3"]
            y-arm vector.
        z : Float[Array, " 3"]
            z-arm vector
        """
        # To be implemented, probably done after my thesis, not important
        raise NotImplementedError

    
    @property
    def vertex(self) -> Float[Array, " 3"]:
        """
        Detector vertex coordinates in the reference celestial frame. Based
        on arXiv:gr-qc/0008066 Eqs. (B11-B13) except for a typo in the
        definition of the local radius; see Section 2.1 of LIGO-T980044-10.

        Returns
        -------
        vertex : Float[Array, " 3"]
            detector vertex coordinates.
        """
        raise NotImplementedError
        
        
    def td_response(
        self,
        T: Float, # total duration
        dt: Float, # time resolution
        t0: Float, # start time, i.e. how much time of the generated waveform will be scrapped
        waveform, # GW class
        wave_parameters: list[Float], # waveform specific parameters
        **kwargs
    )-> Float[Array, " 3 n_sample"]:
        """
        Calculates the time domain response functions for a given GW source (currently only GBs, I'll figure out how to generalise later)
        
        Returns
        --------
        Array with 3 channels (depending on the tdi channel either XYZ or AET)
        
        """
        
        # XYZ Waveform

        tdi_kwargs = dict(
            order=self.order, tdi=self.tdi_gen, tdi_chan=self.channel)

        wrapper = ResponseWrapper(
            waveform,
            T,
            dt,
            index_lambda,
            index_beta,
            t0=t0,
            flip_hx=False,  # set to True if waveform is h+ - ihx
            use_gpu=self.use_gpu,
            remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
            is_ecliptic_latitude=True,  # False if using polar angle (theta)
            remove_garbage=True,  # removes the beginning of the signal that has bad information
            orbits=self.get_orbit(),
            **tdi_kwargs,
        )
    
        chans = wrapper(*wave_parameters)
        
        return jnp.array(chans) #np.array((chan1, chan2, chan3)) # i dont know how you would generalise this to all possible sources


    def fd_response(
        self,
        T: Float, # total duration
        dt: Float, # time resolution
        t0: Float, # start time, i.e. how much time of the generated waveform will be scrapped
        waveform, # GW class
        wave_parameters: list, # waveform specific parameters
        **kwargs
    ) -> Float[Array, " 3 n_sample"]:
        """
        Turns the td-response into a fd-response by just FFTing
        note: you should probably keep the window in mind (to be implemented)
        """
        
        def maskbool(f,armlength, resolution = 1e-2):
            '''
            Masks away all the bad frequencies, i.e. all f \in \Bigcup_{n\in \mathb{N}} [ (-.01+n)/2L , (.01+n)/2L ] 
            NOTE: assumes an approximate constant armlength!
            Input:
            frequency
            
            '''
            
            f = xp.asarray(f)
            print(armlength)
            modes = ((f*armlength-1/(2)) % (1)) -1/(2)
            
            return (xp.abs(modes)) < resolution
        
        
        chans = self.td_response(
            T, # total duration
            dt, # time resolution
            t0, # start time, i.e. how much time of the generated waveform will be scrapped
            waveform, # GW class
            wave_parameters, # waveform specific parameters
            **kwargs)
        
        response = jnp.fft.rfft(chans) # add window to this to avoid Gibbs phenomena
        freqs    = jnp.fft.rfftfreq(len(chans[0]), d = dt) # easiest way to get the correct frequencies
        
        if self.orbit == 'equal':
            orbitclass = self.get_orbit()
            L = orbitclass.get_light_travel_times(0.0, 12)
            mask_array = jnp.array(1-maskbool(freqs, L))
            response = response * mask_array
        elif self.orbit == 'ESA':
            raise NotImplementedError
            
        
        if kwargs['with_freqs'] ==True:
            return jnp.array(response), freqs
        
        return jnp.array(response)

    
    
    def inject_signal(
        self,
        key: PRNGKeyArray,
        freqs: Float[Array, " n_sample"],
        waveform, # waveform class of the source
        params: dict, # contains important noise parameters Pij 
        
    ) -> None:
        
        def PSDs_to_covariance(PSD):
            # Computes the covariance matrix such that ( Re[n(f)] , Im[n(f)]  ) ~ N(0, bigPSD)
            # when sampling noise
            # Only relevant when sampling the real and imaginary parts of the noise!
            
            '''
            bigPSD = [ Re{PSD} -Im{PSD} ]
                     [ Im{PSD}  Re{PSD} ]
            '''
    
            realPSD  = xp.real(PSD)
            imagPSD  = xp.imag(PSD)
    
            diag     = xp.eye(2) # real components
            antidiag = xp.array([[0,-1],[1,0]]) # imaginary components
    
            bigPSD = xp.kron(diag,realPSD) + xp.kron(antidiag,imagPSD) 
    
            return bigPSD

        freqs, signals     = fd_response(params['T'], params['dt'], params['t0']  , {'with_freqs':True})
        
        # symmetric noise curves
        where_freqs = (freqs>0) 
        freqs = freqs[where_freqs]
        
        self.frequencies = freqs
        
        psds, csds = self.load_psds(freqs[where_freqs], params['Pij'], params['Aij'], params['Lij'])
        
        
        
        psd =  jnp.array([
            [psds[0].real  , csds[0]       , csds[2].conj()],
            [csds[0].conj(), psds[1].real  , csds[1]       ],
            [csds[2]       , csds[1].conj(), psds[2].real  ]])
        
        psd = jnp.einsum('ijk->kij', psd)
        
        cov = psd / (4 * (freqs[1] - freqs[0]))
        
        
        self.psd = psd
        
        print('Sampling the noise...')
        mean = jnp.zeros(shape = freqs.shape + (6,))
        separated_noises = jax.random.multivariate_normal(key, mean, PSDs_to_covariance(cov) /2 , shape = freqs.shape)
        #noises_re = jax.random.multivariate_normal(key, mean, cov / 2, shape=freqs.shape)
        
        noises = separated_noise.T[:3] + 1j*separated_noise.T[3:]   # connecting real part of the noise and the imaginary part into a complex array
        print('done!')
        
        print('Adding noise to the signals...')
        self.data = jnp.array([
            signals[0][where_freqs] + noises[0],
            signals[1][where_freqs] + noises[1],
            signals[2][where_freqs] + noises[2]
        ]).T
        print('done!')
        
        print('Calculating the SNRs')
        
        inv_cov          = jnp.linalg.inv(cov) #inv_3x3_matrix(cov_sym)
    
        optimal_SNR_2    = jnp.einsum('ij,ijk,ik->',
                              signals,
                              inv_cov,
                              signals.conj() ).real
    
        optimal_SNR      = jnp.sqrt(optimal_SNR_2)
        
        match_filter_SNR = jnp.einsum(
            'ij,ijk,ik->',
            self.data,
            inv_cov,
            signals.conj(),
        )
        match_filter_SNR /= optimal_SNR
        
        print(f"For {self.name}'s channels ({self.tdi_channel}):")
        print(f"The correlated optimal SNR is {optimal_SNR}")
        print(f"The correlated match filter SNR is {match_filter_SNR}")
        
        
    @jaxtyped
    def load_psds(self,f_arr, Parr, Aarr, Larr):
        '''
        Loads in the Covariance matrix for a given set of noise parameters 
            Parr: array of the OMS noise paramaters,
            Aarr: array of the TM noise parameters,
            Larr: array of the (constant) armlengths,
        Currently, I only have 1st gen TDI programmed and for only XYZ and AET. Might add Sagnac variables later.
        
        Returns:
            Array of shape: #freqs x 3 x 3
        
        '''
        
        XXYYZZ = list()
        XYYZZX = list()
        
        
        def Partial_PSDs(f,i,j): # partial sums of PSD
        
            S_ij_ij = S_ij_OMS(f, P = Parr[i,j] ) + S_ij_TM(f, A=Aarr[i,j] ) + S_ij_TM(f, A=Aarr[j,i] )
            S_ji_ji = S_ij_OMS(f, P = Parr[j,i] ) + S_ij_TM(f, A=Aarr[j,i] ) + S_ij_TM(f, A=Aarr[i,j] )
            S_ij_ji = xp.exp(2j*xp.pi*f*Lij[j,i]) * S_ij_TM(f, A=Aarr[i,j] ) + xp.exp(-2j*xp.pi*f*Lij[i,j]) * S_ij_TM(f, A=Aarr[j,i])
            S_ji_ij = xp.exp(2j*xp.pi*f*Lij[i,j]) * S_ij_TM(f, A=Aarr[j,i] ) + xp.exp(-2j*xp.pi*f*Lij[j,i]) * S_ij_TM(f, A=Aarr[i,j])
        
            return S_ij_ij, S_ji_ji, S_ij_ji, S_ji_ij
        
        ''' Idk whether I'll explicitly note this down, probably should later for better numerical accuracy/stability but oh well
        
        if noise_equal:
            S_XX =  16*xp.sin(2*xp.pi*f_arr*Larr[0,1])**2 * ( S_ij_OMS(f_arr, P = Parr[0,1] ) + 3*S_ij_TM(f_arr, A=Aarr[0,1]) )
            
            S_XY =  -4*xp.sin(2*xp.pi*f_arr*Larr[0,1])*xp.sin(4*xp.pi*f_arr*Larr[0,1]) * (S_ij_OMS(f_arr, P = Parr[0,1] ) + 4*S_ij_TM(f_arr, A=Aarr[0,1]) )
            
            XXYYZZ = [S_XX]*3
            XYYZZX = [S_XY]*3
            
            return XXYYZZ, XYYZZX
        '''
                
        dimension = 3
        for k in range(dimension): #XYZ column
            S12 = Partial_PSDs(f_arr,k,(k+1)%dimension)
            S13 = Partial_PSDs(f_arr,k,(k+2)%dimension)
            
            D12 = D_ij(f_arr,k,(k+1)%dimension)
            D21 = D_ij(f_arr,(k+1)%dimension,k)
            D13 = D_ij(f_arr,k,(k+2)%dimension)
            D31 = D_ij(f_arr,(k+2)%dimension,k)
            D23 = D_ij(f_arr,(k+1)%dimension,(k+2)%dimension)
            D32 = D_ij(f_arr,(k+2)%dimension,(k+1)%dimension)
            
            term1 = xp.abs(1 - D13 * D31)**2 *(S12[0] + xp.abs(D12)**2*S12[1] + D12*S12[2] + S12[3]*xp.conj(D21))
            term2 = xp.abs(1 - D12 * D21)**2 *(S13[0] + xp.abs(D13)**2*S13[1] + D13*S13[2] + S13[3]*xp.conj(D31))
            
            csdterm = (1-xp.conj(D13*D31))*(D23*D32-1)*(D12*S12[0] + xp.conj(D21)*S12[1] + S12[2] + D21*xp.conj(D12)*S12[3])
            
            XXYYZZ.append(term1+term2)
            XYYZZX.append(csdterm)
            
        if self.channel = 'XYZ':
            return XXYYZZ, XYYZZX
        elif self.channel = 'AET':
            
            XX, YY, ZZ = XXYYZZ
            XY, YZ, ZX = XYYZZX
            
            AA       = ( ZZ + XX   - 2*ZX.real)/2
            EE       = ( XX + 4*YY + ZZ - 4 * (XY + YZ - ZX/2).real ) / 6
            TT       = ( XX + YY   + ZZ + 2 * (XY + YZ + ZX  ).real ) / 3 
            AE       = ( ZZ - XX   + 2* ZX.imag + 2*(XY-YZ) ) / xp.sqrt(12)
            ET       = ( XX - 2*YY + ZZ  + XY   - 2*XY.conj() + 2*ZX.real + YZ.conj() - 2*YZ ) / xp.sqrt(18)
            TA       = ( ZZ - XX   + 2*ZX.imag  - XY.conj()   + YZ)     
            
            AAEETT = [AA,EE,TT]
            AEETTA = [AE,ET,TA]
            return AAEETT, AEETTA
            

'''
def maskbool(f,armlength):
    f = xp.asarray(f)
    print(armlength)
        modes = ((f*armlength-1/(2)) % (1)) -1/(2)
    
    return (xp.abs(modes)) < 1e-2
'''







######################################################################################################
################################# ET DETECTOR CLASS FOR REFERENCE ####################################
######################################################################################################

    class TriangularGroundBased3G(Detector):
    polarization_mode: list[Polarization]
    frequencies: Float[Array, " n_sample"]
    data: Float[Array, " n_sample 3"]
    psd: Float[Array, " n_sample 3 3"]

    # giving the values for one of the three detectors
    # the rest will be automatically calculated
    latitude: Float = 0
    longitude: Float = 0
    xarm_azimuth: Float = 0
    yarm_azimuth: Float = 0
    xarm_tilt: Float = 0
    yarm_tilt: Float = 0
    elevation: Float = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name

        self.latitude = kwargs.get("latitude", 0)
        self.longitude = kwargs.get("longitude", 0)
        self.elevation = kwargs.get("elevation", 0)
        self.xarm_azimuth = kwargs.get("xarm_azimuth", 0)
        self.yarm_azimuth = kwargs.get("yarm_azimuth", 0)
        self.xarm_tilt = kwargs.get("xarm_tilt", 0)
        self.yarm_tilt = kwargs.get("yarm_tilt", 0)
        modes = kwargs.get("mode", "pc")

        self.polarization_mode = [Polarization(m) for m in modes]
        self.frequencies = jnp.array([])
        self.data = jnp.array([])
        self.psd = jnp.array([])

    @staticmethod
    def _get_arm(
        lat: Float, lon: Float, tilt: Float, azimuth: Float
    ) -> Float[Array, " 3"]:
        """
        Construct detector-arm vectors in Earth-centric Cartesian coordinates.

        Parameters
        ---------
        lat : Float
            vertex latitude in rad.
        lon : Float
            vertex longitude in rad.
        tilt : Float
            arm tilt in rad.
        azimuth : Float
            arm azimuth in rad.

        Returns
        -------
        arm : Float[Array, " 3"]
            detector arm vector in Earth-centric Cartesian coordinates.
        """
        e_lon = jnp.array([-jnp.sin(lon), jnp.cos(lon), 0])
        e_lat = jnp.array(
            [-jnp.sin(lat) * jnp.cos(lon), -jnp.sin(lat) * jnp.sin(lon), jnp.cos(lat)]
        )
        e_h = jnp.array(
            [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)]
        )

        return (
            jnp.cos(tilt) * jnp.cos(azimuth) * e_lon
            + jnp.cos(tilt) * jnp.sin(azimuth) * e_lat
            + jnp.sin(tilt) * e_h
        )

    @property
    def arms(self) -> tuple[Float[Array, " 3"],
                            Float[Array, " 3"],
                            Float[Array, " 3"]
                            ]:
        """
        Detector arm vectors (x, y, z).

        Returns
        -------
        x : Float[Array, " 3"]
            x-arm vector.
        y : Float[Array, " 3"]
            y-arm vector.
        z : Float[Array, " 3"]
            z-arm vector
        """
        x = self._get_arm(
            self.latitude, self.longitude, self.xarm_tilt, self.xarm_azimuth
        )
        y = self._get_arm(
            self.latitude, self.longitude, self.yarm_tilt, self.yarm_azimuth
        )
        z = y - x
        return x, y, z

    @staticmethod
    def _get_tensor(arm1, arm2) -> Float[Array, " 3 3"]:
        """
        Detector tensor defining the strain measurement.

        Returns
        -------
        tensor : Float[Array, " 3 3"]
                 detector tensor.
        """

        return 0.5 * (
            jnp.einsum("i,j->ij", arm1, arm1) - jnp.einsum("i,j->ij", arm2, arm2)
        )

    @property
    def tensors(self) -> tuple[Float[Array, " 3 3"],
                               Float[Array, " 3 3"],
                               Float[Array, " 3 3"]
                               ]:
        """
        Detector tensor defining the strain measurement.

        Returns
        -------
        D1_tensor : Float[Array, " 3 3"]
                    D1 detector tensor.
        D2_tensor : Float[Array, " 3 3"]
                    D2 detector tensor.
        D3_tensor : Float[Array, " 3 3"]
                    D3 detector tensor.
        """
        x, y, z = self.arms

        D1 = self._get_tensor(x, y)
        D2 = self._get_tensor(z, -x)
        D3 = self._get_tensor(-y, -z)

        return D1, D2, D3

    @property
    def vertex(self) -> Float[Array, " 3"]:
        """
        Detector vertex coordinates in the reference celestial frame. Based
        on arXiv:gr-qc/0008066 Eqs. (B11-B13) except for a typo in the
        definition of the local radius; see Section 2.1 of LIGO-T980044-10.

        Returns
        -------
        vertex : Float[Array, " 3"]
            detector vertex coordinates.
        """
        # get detector and Earth parameters
        lat = self.latitude
        lon = self.longitude
        h = self.elevation
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2 * (
            major**2 * jnp.cos(lat) ** 2 + minor**2 * jnp.sin(lat) ** 2
        ) ** (-0.5)
        x = (r + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (r + h) * jnp.cos(lat) * jnp.sin(lon)
        z = ((minor / major) ** 2 * r + h) * jnp.sin(lat)
        return jnp.array([x, y, z])

    def delay_from_geocenter(self, ra: Float, dec: Float, gmst: Float) -> Float:
        """
        Calculate time delay between two detectors in geocentric
        coordinates based on XLALArrivaTimeDiff in TimeDelay.c

        https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_delay__h.html

        Parameters
        ---------
        ra : Float
            right ascension of the source in rad.
        dec : Float
            declination of the source in rad.
        gmst : Float
            Greenwich mean sidereal time in rad.

        Returns
        -------
        Float: time delay from Earth center.
        """
        delta_d = -self.vertex
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec
        omega = jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )
        return jnp.dot(omega, delta_d) / C_SI

    def antenna_pattern(self, ra: Float, dec: Float, psi: Float, gmst: Float) -> list:
        """
        Computes {name} antenna patterns for {modes} polarizations
        at the specified sky location, orientation and GMST.

        In the long-wavelength approximation, the antenna pattern for a
        given polarization is the dyadic product between the detector
        tensor and the corresponding polarization tensor.

        Parameters
        ---------
        ra : Float
            source right ascension in radians.
        dec : Float
            source declination in radians.
        psi : Float
            source polarization angle in radians.
        gmst : Float
            Greenwich mean sidereal time (GMST) in radians.
        modes : str
            string of polarizations to include, defaults to tensor modes: 'pc'.

        Returns
        -------
        result : list
            antenna pattern values for {modes}.
        """
        D1, D2, D3 = self.tensors

        antenna_patterns = []
        for D in [D1, D2, D3]:
            antenna_patterns_per_det = {}
            for polarization in self.polarization_mode:
                wave_tensor = polarization.tensor_from_sky(ra, dec, psi, gmst)
                antenna_patterns_per_det[polarization.name] = jnp.einsum(
                    "ij,ij->", D, wave_tensor
                )
            antenna_patterns.append(antenna_patterns_per_det)

        return antenna_patterns

    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict[str, Float],
        **kwargs,
    ) -> Float[Array, " 3 n_sample"]:
        """
        Modulate the waveform in the sky frame by the detector response in the time domain.
        """
        raise NotImplementedError

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict[str, Float],
        **kwargs,
    ) -> Float[Array, " 3 n_sample"]:
        """
        Modulate the waveform in the sky frame by the detector response in the frequency domain.
        """
        ra, dec, psi, gmst = params["ra"], params["dec"], params["psi"], params["gmst"]
        antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
        timeshift = self.delay_from_geocenter(ra, dec, gmst)

        response = []

        for antenna_pattern_per_det in antenna_pattern:
            h_detector = jax.tree_util.tree_map(
                lambda h, antenna: h
                * antenna
                * jnp.exp(-2j * jnp.pi * frequency * timeshift),
                h_sky,
                antenna_pattern_per_det,
            )
            response.append(jnp.sum(jnp.stack(jax.tree_util.tree_leaves(h_detector)), axis=0))
        return jnp.array(response).T

    def inject_signal(
        self,
        key: PRNGKeyArray,
        freqs: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict[str, Float],
        psd_file_dict: dict = {'XX': '', 'YY': '', 'ZZ': ''},
        csd_file_dict: dict = {'XY': '', 'E23': '', 'XZ': ''},
    ) -> None:
        """
        Inject a signal into the detector data.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX PRNG key.
        freqs : Float[Array, " n_sample"]
            Array of frequencies.
        h_sky : dict[str, Float[Array, " n_sample"]]
            Array of waveforms in the sky frame. The key is the polarization mode.
        params : dict[str, Float]
            Dictionary of parameters.
        psd_file_dict : dict
            Dictionary with paths to the PSD files, with keys E1, E2, E3
        csd_file_dict : dict
            Dictionary with paths to the CSD files, with keys E12, E23, E13

        Returns
        -------
        None
        """
        self.frequencies = freqs
        E1_psd = self.load_psd(freqs, psd_file_dict['E1'])
        E2_psd = self.load_psd(freqs, psd_file_dict['E2'])
        E3_psd = self.load_psd(freqs, psd_file_dict['E3'])

        E12_csd = self.load_csd(freqs, csd_file_dict['E12'])
        E23_csd = self.load_csd(freqs, csd_file_dict['E23'])
        E13_csd = self.load_csd(freqs, csd_file_dict['E13'])

        psd = jnp.array([[E1_psd, E12_csd, E13_csd],
                         [E12_csd.conj(), E2_psd, E23_csd],
                         [E13_csd.conj(), E23_csd.conj(), E3_psd]])
        self.psd = jnp.einsum('ijk->kij', psd)
        cov = self.psd / (4 * (freqs[1] - freqs[0]))

        mean = jnp.zeros(shape=freqs.shape + (3,))

        key, subkey = jax.random.split(key, 2)

        noises_re = jax.random.multivariate_normal(key, mean, cov / 2, shape=freqs.shape)
        noises_im = jax.random.multivariate_normal(subkey, mean, cov / 2, shape=freqs.shape)
        noises = noises_re + 1j * noises_im
        align_time = 1
        '''jnp.exp(
            -1j * 2 * jnp.pi * freqs * (params["epoch"] + params["t_c"])
        )'''

        signals = self.fd_response(freqs, h_sky, params)
        self.data = jnp.array([signals.T[0] * align_time + noises.T[0],
                               signals.T[1] * align_time + noises.T[1],
                               signals.T[2] * align_time + noises.T[2]]).T

        # calculate the optimal SNR and match filter SNR
        inv_cov = jnp.linalg.inv(cov)

        optimal_SNR_2 = jnp.einsum(
            'ij,ijk,ik->',
            signals,
            inv_cov,
            signals.conj()
        ).real
        optimal_SNR = jnp.sqrt(optimal_SNR_2)

        match_filter_SNR = jnp.einsum(
            'ij,ijk,ik,i->',
            self.data,
            inv_cov,
            signals.conj(),
            align_time.conj(),
        )
        match_filter_SNR /= optimal_SNR

        print(f"For detector {self.name}:")
        print(f"The correlated Optimal SNR is {optimal_SNR}")
        print(f"The correlated match filter SNR is {match_filter_SNR}")

        cov_uncorr = cov * jnp.einsum('i,jk->ijk', jnp.ones(freqs.shape), jnp.eye(3))
        inv_cov_uncorr = jnp.linalg.inv(cov_uncorr)

        optimal_SNR_2 = jnp.einsum(
            'ij,ijk,ik->',
            signals,
            inv_cov_uncorr,
            signals.conj()
        ).real
        optimal_SNR = jnp.sqrt(optimal_SNR_2)

        match_filter_SNR = jnp.einsum(
            'ij,ijk,ik,i->',
            self.data,
            inv_cov_uncorr,
            signals.conj(),
            align_time.conj(),
        )
        match_filter_SNR /= optimal_SNR

        print(f"The uncorrelated Optimal SNR is {optimal_SNR}")
        print(f"The uncorrelated match filter SNR is {match_filter_SNR}")

    @jaxtyped
    def load_psd(
        self, freqs: Float[Array, " n_sample"], psd_file: str = ""
    ) -> Float[Array, " n_sample"]:
        if psd_file == "":
            print("Grabbing ET-D PSD")
            url = psd_file_dict["ET-D"]
            data = requests.get(url)
            open(self.name + ".txt", "wb").write(data.content)
            f, psd_vals = np.loadtxt(self.name + ".txt", unpack=True)
        else:
            f, psd_vals = np.loadtxt(psd_file, unpack=True)

        psd = interp1d(f, psd_vals, fill_value=(psd_vals[0], psd_vals[-1]))(freqs)  # type: ignore
        psd = jnp.array(psd)
        return psd

    @jaxtyped
    def load_csd(
        self, freqs: Float[Array, " n_sample"], csd_file: str = ""
    ) -> Float[Array, " n_sample"]:
        if csd_file == "":
            print("Assuming no correlation in noise")
            csd_vals = jnp.zeros(shape=freqs.shape)
            f = freqs
        else:
            f, csd_vals = np.loadtxt(csd_file, unpack=True)

        csd = interp1d(f, csd_vals, fill_value=(csd_vals[0], csd_vals[-1]))(freqs)  # type: ignore
        csd = jnp.array(csd)
        return csd
