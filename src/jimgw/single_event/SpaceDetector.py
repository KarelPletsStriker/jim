#!/usr/bin/env python

"""
LISA code written by Karel Plets - Implementing the SpaceBased Detector Class
Date: 28/03/2025

Notes:  ---------- CUMULATIVE VERSION-----------------
        
"""

############################################
#############  IMPORT PACKAGES #############
############# DEFINE VARIABLES #############
############################################

# LISA RELEVANT PACKAGES

import numpy as np
import h5py
from astropy import units as un

from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
from jimgw.jaxlisaresponse.response import pyResponseTDI, ResponseWrapper

equal = EqualArmlengthOrbits()
equal.configure(linear_interp_setup=True)

import time
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Float, jaxtyped


from jimgw.jim import Jim
from jimgw.single_event.detector import Detector
from jimgw.single_event.wave import Polarization
from jimgw.single_event.waveform import Waveform








def S_ij_TM(f, A = 1):
        
    # write everything in SI units
    c = 299792458 #m/s
    pi = np.pi
    
    S = A**2 * 1e-30 * (1 + ( 4e-4 / f )**2) * (1 + (f/8e-3)**4) / (2*pi*c*f)**2

    return S

# Optical Metrology System (OMS) noise PSD
def S_ij_OMS(f, P = 1):
        
    # write everything in SI units
    c = 299792458 #m/s
    pi = np.pi
    
    S = P**2 *1e-24 * (1 + ( 2e-3 / f )**4) * (2*pi*f/c)**2

    return S





class SpaceBased(Detector):
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
    
    

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __init__(self, name: str, **kwargs) -> None: # To be changed into stuff you need for FD analysis
        self.name = name
        
        modes = kwargs.get("mode", "pc")
        


        '''
        I think these are useless parameters for LISA?
        
        self.latitude = kwargs.get("latitude", 0)
        self.longitude = kwargs.get("longitude", 0)
        self.elevation = kwargs.get("elevation", 0)
        self.xarm_azimuth = kwargs.get("xarm_azimuth", 0)
        self.yarm_azimuth = kwargs.get("yarm_azimuth", 0)
        self.xarm_tilt = kwargs.get("xarm_tilt", 0)
        self.yarm_tilt = kwargs.get("yarm_tilt", 0)
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
        self.src_env = kwargs.get('src_env','/data/leuven/347/vsc34717/python/miniconda3/envs/lisa102/lib/python3.10/site-packages/')
        self.order = kwargs.get('order',25) # order of lagrangian interpolation, fastlisaresponse parameter
        
        self.detector_parameters = {
            'T' : kwargs.get('T', 2),  # Duration of the simulation, in years
            'dt' : kwargs.get('dt', 2.5),
            't0' : kwargs.get('t0', 10000.0), # time at which signal starts (chops off data at start of waveform where information is not correct)
            'index_beta' : kwargs.get('index_beta',7),
            'index_lambda' : kwargs.get('index_lambda',6)
        }

        self.SNR_method = kwargs.get('SNR_method', 'masking') # other options are regularization, smoothing (to be added) and None

        if self.SNR_method == 'masking':
            self.masking_resolution = kwargs.get('masking_resolution', 1e-2)
        
    
    def get_orbit(self):
        
        if self.orbit == "equal":
            #orbit = EqualArmlengthOrbits(use_gpu = self.use_gpu)
            #orbit.configure(linear_interp_setup=True)

            orbit = '/jimgw/single_event/orbits/equalarmlength-trailing-fit.h5'
            return self.src_env+orbit
        elif self.orbit == "ESA":
            #orbit = ESAOrbits(use_gpu = self.use_gpu)
            #orbit.configure(linear_interp_setup=True)

            orbit = '/jimgw/single_event/orbits/esa-trailing-orbits.h5'
            return self.src_env+orbit
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
        waveform: Waveform, # GW class
        detector_parameters : dict[Float], # Simulation Parameters: 'T' (total duration), 't0' (start time, mostly to scrap shitty data), 'dt' (time resolution)
        # 'index_lambda' (), 'index_beta' ()
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
            order=self.order, 
            tdi=self.tdi_gen,
            tdi_chan=self.channel,
            orbit_kwargs = dict(orbit_file=self.get_orbit()))

        wrapper = ResponseWrapper(
            waveform,
            detector_parameters['T'],
            detector_parameters['dt'],
            detector_parameters['index_lambda'],
            detector_parameters['index_beta'],
            t0=detector_parameters['t0'],
            flip_hx=False,  # set to True if waveform is h+ - ihx
            use_gpu=self.use_gpu,
            remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
            is_ecliptic_latitude=True,  # False if using polar angle (theta)
            remove_garbage=True,  # removes the beginning of the signal that has bad information
            #orbits=self.get_orbit(),
            **tdi_kwargs,
        )
        
        chans = wrapper(*wave_parameters)
        
        return jnp.array(chans) #np.array((chan1, chan2, chan3)) # i dont know how you would generalise this to all possible sources


    def fd_response(
        self,
        waveform: Waveform, # GW class
        params: dict[Float], # waveform specific parameters
        **kwargs
    ) -> Float[Array, " 3 n_sample"]:
        """
        Turns the td-response into a fd-response by just FFTing
        note: you should probably keep the window in mind (to be implemented)
        """
        
        
        
        wave_parameters = [
            params['A'],
            params['f'],
            params['fdot'],
            params['iota'],
            params['phi0'],
            params['psi'],
            params['lam'],
            params['beta']
        ]

        
        chans = self.td_response(
            waveform, # GW class
            self.detector_parameters,
            wave_parameters, # waveform specific parameters
            **kwargs)
        
        response = jnp.fft.rfft(chans) # add window to this to avoid Gibbs phenomena
        freqs    = jnp.fft.rfftfreq(len(chans[0]), d = self.detector_parameters['dt'])
        
        '''
        if self.orbit == 'equal':
            orbitclass = equal
            L = orbitclass.get_light_travel_times(0.0, 12)
            
            mask_array = jnp.array(maskbool(freqs, L, resolution = 0))
            response   = [jnp.where(mask_array,chan, jnp.zeros(len(chan))) for chan in response]
            #freqs      = jnp.where(mask_array,freqs)
            
        elif self.orbit == 'ESA':
            raise NotImplementedError'''
            
        
        if kwargs.get('with_freqs', False) == True:
            # easiest way to get the correct frequencies
            return jnp.array(response), freqs
        
        return jnp.array(response)

    

    def calc_SNR(
        self,
        x,
        y,
        ):
        '''
        This function calculates the SNR between 2 arrays for a space detector: (array)

        parameters:
        x - left vector, has shape (num_channels, num_freqs)
        y - right vector, has shape (num_channels, num_freqs)
        '''
        cov = self.psd / (4 * (self.frequencies[1] - self.frequencies[0]))
        inv_cov = jnp.linalg.inv(cov)


        def maskbool(f,armlength, resolution = 1e-2):
            '''
            Masks away all the bad frequencies, i.e. all f \in \Bigcup_{n\in \mathb{N}} [ (-.01+n)/2L , (.01+n)/2L ] 
            NB: assumes an approximate constant armlength!
            Input:
            frequency

            Returns:
            Boolean array with all the excluded frequencies set to False, rest is True
            '''
            
            f = jnp.asarray(f)
            #print(armlength)
            modes = ((f*armlength+1/2) % (1)) - 1/(2)
            
            return (jnp.abs(modes)) > resolution
        
        if self.SNR_method == 'masking':

            mask_array = maskbool(self.frequencies, self.armlength, resolution = self.masking_resolution)

            res = jnp.einsum('ji,ijk,ki->i', x, inv_cov, y.conj()).real
            return jnp.sum(res, where = mask_array)

        elif self.SNR_method == 'smoothing':
            raise NotImplementedError

        elif (self.SNR_method == None) or (self.SNR_method == 'regularization'):
            return jnp.einsum('ji,ijk,ki->', x, inv_cov, y.conj()).real

        else:
            raise NotImplementedError




            

        

    
    
    
    def inject_signal(
        self,
        key: PRNGKeyArray,
        #freqs: Float[Array, " n_sample"],
        waveform: Waveform, # waveform class of the source 
        params: dict, # contains important noise parameters Aij, Pij, Lij
        
    ) -> None:
        
        def PSDs_to_covariance(PSD):
            # Computes the covariance matrix such that ( Re[n(f)] , Im[n(f)]  ) ~ N(0, bigPSD)
            # when sampling noise
            # Only relevant when sampling the real and imaginary parts of the noise!
            
            '''
            bigPSD = [ Re{PSD} -Im{PSD} ]
                     [ Im{PSD}  Re{PSD} ]
            '''
    
            realPSD  = jnp.real(PSD)
            imagPSD  = jnp.imag(PSD)
    
            diag     = jnp.eye(2) # real components
            antidiag = jnp.array([[0,-1],[1,0]]) # imaginary components
    
            bigPSD = jnp.kron(diag,realPSD) + jnp.kron(antidiag,imagPSD) 
    
            return bigPSD

        signals, freqs     = self.fd_response(waveform, params , with_freqs = True)
        
        # symmetric noise curves
        
        self.frequencies = freqs
        self.signals     = signals
        
        psds, csds = self.load_psds(self.frequencies, params['Pij'], params['Aij'], params['Lij'])
        
        print(psds[0].shape, csds[0].shape)
        
        psd =  jnp.array([
            [psds[0].real  , csds[0]       , csds[2].conj()],
            [csds[0].conj(), psds[1].real  , csds[1]       ],
            [csds[2]       , csds[1].conj(), psds[2].real  ]
        ]) 
        
        psd = jnp.einsum('ijk->kij', psd)
        
        self.psd = psd
        
        
        cov = psd / 4 / (self.frequencies[1] - self.frequencies[0])
        print('Sampling the noise...')
        
        mean = jnp.zeros(shape = self.frequencies.shape + (6,))
        separated_noise = jax.random.multivariate_normal(key, mean, PSDs_to_covariance(cov) /2 , shape = self.frequencies.shape)
        print('noise real nans:',jnp.sum(jnp.isnan(separated_noise.T[:3])))
        print('noise imag nans:',jnp.sum(jnp.isnan(separated_noise.T[3:])))
        noises = separated_noise.T[:3] + 1j*separated_noise.T[3:]   # connecting real part of the noise and the imaginary part into a complex array
        noises = jnp.where(jnp.isnan(noises), 0, noises)
        print('done!')
        
        print('Adding noise to the signals...')

        self.data = jnp.array([
            signals[0] + noises[0],
            signals[1] + noises[1],
            signals[2] + noises[2]
        ])
        
        print('done!')
        
        print('Calculating the SNRs')
        
        inv_cov          = jnp.linalg.inv(cov) #inv_3x3_matrix(cov_sym)
        #print(signals.shape, inv_cov.shape)
        optimal_SNR_2    = self.calc_SNR(
                              signals,
                              signals,
                              )
    
        optimal_SNR      = jnp.sqrt((optimal_SNR_2))
        
        match_filter_SNR = self.calc_SNR(self.data, signals)
        
        
        match_filter_SNR =  match_filter_SNR / optimal_SNR
        
        
        print(f"For {self.name}'s channels ({self.channel}):")
        print(f"The correlated optimal SNR is {optimal_SNR}")
        print(f"The correlated match filter SNR is {match_filter_SNR}")
        
        
    def load_psds(self,f_arr, Parr, Aarr, Lij):
        '''
        Loads in the Covariance matrix for a given set of noise parameters 
            Parr: array of the OMS noise paramaters,
            Aarr: array of the TM noise parameters,
            Lij: array of the (constant) armlengths,
        Currently, I only have 1st gen TDI programmed and for only XYZ and AET. Might add Sagnac variables later.
        
        Returns:
            Array of shape: #(freqs) x 3 x 3
        
        '''
        def D_ij(f,i,j):
            return np.exp(-2j*np.pi*f*Lij[i,j])

        
        XXYYZZ = list()
        XYYZZX = list()

        epsilon = 0
        if self.SNR_method == 'masking':
            self.armlength = Lij[0,1]
        elif self.SNR_method == 'regularization':
            epsilon += 1e-42

        
        
        def Partial_PSDs(f,i,j): # partial sums of PSD
        
            S_ij_ij = S_ij_OMS(f, P = Parr[i,j] ) + S_ij_TM(f, A=Aarr[i,j] ) + S_ij_TM(f, A=Aarr[j,i] )
            S_ji_ji = S_ij_OMS(f, P = Parr[j,i] ) + S_ij_TM(f, A=Aarr[j,i] ) + S_ij_TM(f, A=Aarr[i,j] )
            S_ij_ji = jnp.exp(2j*jnp.pi*f*Lij[j,i]) * S_ij_TM(f, A=Aarr[i,j] ) + jnp.exp(-2j*jnp.pi*f*Lij[i,j]) * S_ij_TM(f, A=Aarr[j,i])
            S_ji_ij = jnp.exp(2j*jnp.pi*f*Lij[i,j]) * S_ij_TM(f, A=Aarr[j,i] ) + jnp.exp(-2j*jnp.pi*f*Lij[j,i]) * S_ij_TM(f, A=Aarr[i,j])
        
            return S_ij_ij, S_ji_ji, S_ij_ji, S_ji_ij
        
        ''' Idk whether I'll explicitly note this down, probably should later for better numerical accuracy/stability but oh well
        
        if noise_equal:
            S_XX =  16*jnp.sin(2*jnp.pi*f_arr*Lij[0,1])**2 * ( S_ij_OMS(f_arr, P = Parr[0,1] ) + 3*S_ij_TM(f_arr, A=Aarr[0,1]) )
            
            S_XY =  -4*jnp.sin(2*jnp.pi*f_arr*Lij[0,1])*jnp.sin(4*jnp.pi*f_arr*Lij[0,1]) * (S_ij_OMS(f_arr, P = Parr[0,1] ) + 4*S_ij_TM(f_arr, A=Aarr[0,1]) )
            
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
            
            term1 = jnp.abs(1 - D13 * D31)**2 *(S12[0] + jnp.abs(D12)**2*S12[1] + D12*S12[2] + S12[3]*jnp.conj(D21))
            term2 = jnp.abs(1 - D12 * D21)**2 *(S13[0] + jnp.abs(D13)**2*S13[1] + D13*S13[2] + S13[3]*jnp.conj(D31))
            
            csdterm = (1-jnp.conj(D13*D31))*(D23*D32-1)*(D12*S12[0] + jnp.conj(D21)*S12[1] + S12[2] + D21*jnp.conj(D12)*S12[3])
            
            XXYYZZ.append(term1+term2+epsilon)
            XYYZZX.append(csdterm)
                
        if self.channel == 'XYZ':
                return XXYYZZ, XYYZZX
        elif self.channel == 'AET':
                '''
                This transforms the XYZ base into the AET base
                '''
                
                XX, YY, ZZ = XXYYZZ
                XY, YZ, ZX = XYYZZX
                AA       = ( ZZ + XX   - 2*ZX.real)/2
                EE       = ( XX + 4*YY + ZZ - 4 * (XY + YZ - ZX/2).real ) / 6
                TT       = ( XX + YY   + ZZ + 2 * (XY + YZ + ZX  ).real ) / 3 
                AE       = ( ZZ - XX   + 2* ZX.imag + 2*(XY-YZ) ) / jnp.sqrt(12)
                ET       = ( XX - 2*YY + ZZ  + XY   - 2*XY.conj() + 2*ZX.real + YZ.conj() - 2*YZ ) / jnp.sqrt(18)
                TA       = ( ZZ - XX   + 2*ZX.imag  - XY.conj()   + YZ)     
            
                AAEETT = [AA,EE,TT]
                AEETTA = [AE,ET,TA]
                
                return AAEETT, AEETTA
            
        else:
                raise NotImplementedError

LISA_XYZ_equal = SpaceBased(
        'LISA_XYZ_equal',
        orbit = 'equal',
        channel = 'XYZ',
        tdi_gen = '1st generation',
        use_gpu = True,
        order = 25,
)
            
LISA_AET_equal = SpaceBased(
        'LISA_AET_equal',
        orbit = 'equal',
        channel = 'AET',
        tdi_gen = '1st generation',
        use_gpu = True,
        order = 25,
)

