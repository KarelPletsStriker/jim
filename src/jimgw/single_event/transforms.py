import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped
from astropy.time import Time

from jimgw.single_event.detector import GroundBased2G
from jimgw.transforms import ConditionalBijectiveTransform, BijectiveTransform, NtoNTransform
from jimgw.single_event.utils import (
    m1_m2_to_Mc_q,
    Mc_q_to_m1_m2,
    m1_m2_to_Mc_eta,
    Mc_eta_to_m1_m2,
    q_to_eta,
    eta_to_q,
    ra_dec_to_zenith_azimuth,
    zenith_azimuth_to_ra_dec,
    euler_rotation,
    spin_to_cartesian_spin,
)


@jaxtyped(typechecker=typechecker)
class ComponentMassesToChirpMassMassRatioTransform(BijectiveTransform):
    """
    Transform chirp mass and mass ratio to component masses

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        assert (
            "m_1" in name_mapping[0]
            and "m_2" in name_mapping[0]
            and "M_c" in name_mapping[1]
            and "q" in name_mapping[1]
        )

        def named_transform(x):
            Mc, q = m1_m2_to_Mc_q(x["m_1"], x["m_2"])
            return {"M_c": Mc, "q": q}

        self.transform_func = named_transform

        def named_inverse_transform(x):
            m1, m2 = Mc_q_to_m1_m2(x["M_c"], x["q"])
            return {"m_1": m1, "m_2": m2}

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class ComponentMassesToChirpMassSymmetricMassRatioTransform(BijectiveTransform):
    """
    Transform mass ratio to symmetric mass ratio

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        assert (
            "m_1" in name_mapping[0]
            and "m_2" in name_mapping[0]
            and "M_c" in name_mapping[1]
            and "eta" in name_mapping[1]
        )

        def named_transform(x):
            Mc, eta = m1_m2_to_Mc_eta(x["m_1"], x["m_2"])
            return {"M_c": Mc, "eta": eta}

        self.transform_func = named_transform

        def named_inverse_transform(x):
            m1, m2 = Mc_eta_to_m1_m2(x["M_c"], x["q"])
            return {"m_1": m1, "m_2": m2}

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class MassRatioToSymmetricMassRatioTransform(BijectiveTransform):
    """
    Transform mass ratio to symmetric mass ratio

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        assert "q" == name_mapping[0][0] and "eta" == name_mapping[1][0]

        self.transform_func = lambda x: {"eta": q_to_eta(x["q"])}
        self.inverse_transform_func = lambda x: {"q": eta_to_q(x["eta"])}


@jaxtyped(typechecker=typechecker)
class SkyFrameToDetectorFrameSkyPositionTransform(BijectiveTransform):
    """
    Transform sky frame to detector frame sky position

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    gmst: Float
    rotation: Float[Array, " 3 3"]
    rotation_inv: Float[Array, " 3 3"]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        gps_time: Float,
        ifos: GroundBased2G,
    ):
        super().__init__(name_mapping)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        delta_x = ifos[0].vertex - ifos[1].vertex
        self.rotation = euler_rotation(delta_x)
        self.rotation_inv = jnp.linalg.inv(self.rotation)

        assert (
            "ra" in name_mapping[0]
            and "dec" in name_mapping[0]
            and "zenith" in name_mapping[1]
            and "azimuth" in name_mapping[1]
        )

        def named_transform(x):
            zenith, azimuth = ra_dec_to_zenith_azimuth(
                x["ra"], x["dec"], self.gmst, self.rotation
            )
            return {"zenith": zenith, "azimuth": azimuth}

        self.transform_func = named_transform

        def named_inverse_transform(x):
            ra, dec = zenith_azimuth_to_ra_dec(
                x["zenith"], x["azimuth"], self.gmst, self.rotation_inv
            )
            return {"ra": ra, "dec": dec}

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class GeocentricArrivalTimeToDetectorArrivalTimeTransform(ConditionalBijectiveTransform):
    """
    Transform the geocentric arrival time to detector arrival time

    In the geocentric convention, the arrival time of the signal at the
    center of Earth is gps_time + t_c

    In the detector convention, the arrival time of the signal at the
    detecotr is gps_time + time_delay_from_geo_to_det + t_det

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    gmst: Float
    ifo: GroundBased2G

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        conditional_names: list[str],
        gps_time: Float,
        ifo: GroundBased2G,
    ):
        super().__init__(name_mapping, conditional_names)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        self.ifo = ifo

        assert "t_c" in name_mapping[0] and "t_det" in name_mapping[1]
        assert (
            "ra" in conditional_names
            and "dec" in conditional_names
        )

        def _calc_delay(x):
            ra, dec = x["ra"], x["dec"]
            if hasattr(ra, "shape") and len(ra.shape) > 0:
                delay = self.ifo.delay_from_geocenter(ra[0], dec[0], self.gmst)
            else:
                delay = self.ifo.delay_from_geocenter(ra, dec, self.gmst)
            return delay

        def named_transform(x):
            delay = _calc_delay(x)
            t_det = x["t_c"] + delay
            return {
                "t_det": t_det,
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            delay = _calc_delay(x)
            t_c = x["t_det"] - delay
            return {
                "t_c": t_c,
            }

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(ConditionalBijectiveTransform):
    """
    Transform the geocentric arrival phase to detector arrival phase

    In the geocentric convention, the arrival phase of the signal at the
    center of Earth is phase_c / 2 (in ripple, phase_c is the orbital phase)

    In the detector convention, the arrival phase of the signal at the
    detecotr is phase_det = phase_c / 2 + arg R_det

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    gmst: Float
    ifo: GroundBased2G

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        conditional_names: list[str],
        gps_time: Float,
        ifo: GroundBased2G,
    ):
        super().__init__(name_mapping, conditional_names)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        self.ifo = ifo

        assert "phase_c" in name_mapping[0] and "phase_det" in name_mapping[1]
        assert (
            "ra" in conditional_names
            and "dec" in conditional_names
            and "psi" in conditional_names
            and "iota" in conditional_names
        )

        def _calc_R_det(x):
            ra, dec, psi, iota = x["ra"], x["dec"], x["psi"], x["iota"]
            p_iota_term = (1.0 + jnp.cos(iota) ** 2) / 2.0
            c_iota_term = jnp.cos(iota)

            if hasattr(ra, "shape") and len(ra.shape) > 0:
                antenna_pattern = self.ifo.antenna_pattern(ra[0], dec[0], psi[0], self.gmst)
            else:
                antenna_pattern = self.ifo.antenna_pattern(ra, dec, psi, self.gmst)
            p_mode_term = p_iota_term * antenna_pattern["p"]
            c_mode_term = c_iota_term * antenna_pattern["c"]

            return p_mode_term - 1j * c_mode_term

        def named_transform(x):
            R_det = _calc_R_det(x)
            phase_det = jnp.angle(R_det) + x["phase_c"] / 2.0
            return {
                "phase_det": phase_det % (2.0 * jnp.pi),
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            R_det = _calc_R_det(x)
            phase_c = (-jnp.angle(R_det) + x["phase_det"]) * 2.0
            return {
                "phase_c": phase_c % (2.0 * jnp.pi),
            }

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class DistanceToSNRWeightedDistanceTransform(BijectiveTransform):
    """
    Transform the luminosity distance to network SNR weighted distance

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    gmst: Float
    ifos: list[GroundBased2G]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        gps_time: Float,
        ifos: list[GroundBased2G],
    ):
        super().__init__(name_mapping)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        self.ifos = ifos

        assert "d_L" in name_mapping[0] and "d_hat" in name_mapping[1]

        def _calc_R_dets(x):
            ra, dec, psi, iota = (
                x["ra"],
                x["dec"],
                x["psi"],
                x["iota"],
            )
            p_iota_term = (1.0 + jnp.cos(iota) ** 2) / 2.0
            c_iota_term = jnp.cos(iota)
            R_dets2 = 0.0
            for ifo in self.ifos:
                antenna_pattern = ifo.antenna_pattern(ra, dec, psi, self.gmst)
                p_mode_term = p_iota_term * antenna_pattern["p"]
                c_mode_term = c_iota_term * antenna_pattern["c"]
                R_dets2 += p_mode_term**2 + c_mode_term**2

            return jnp.sqrt(R_dets2)

        def named_transform(x):
            d_L, M_c = (
                x["d_L"],
                x["M_c"],
            )
            R_dets = _calc_R_dets(x)
            d_hat = d_L / jnp.power(M_c, 5.0 / 6.0) / R_dets
            return {
                "d_hat": d_hat,
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            d_hat, M_c = (
                x["d_hat"],
                x["M_c"],
            )
            R_dets = _calc_R_dets(x)
            d_L = d_hat * jnp.power(M_c, 5.0 / 6.0) * R_dets
            return {
                "d_L": d_L,
            }

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class SpinToCartesianSpinTransform(NtoNTransform):
    """
    Spin to Cartesian spin transformation
    """

    freq_ref: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        freq_ref: Float,
    ):
        super().__init__(name_mapping)

        self.freq_ref = freq_ref

        assert (
            "theta_jn" in name_mapping[0]
            and "phi_jl" in name_mapping[0]
            and "theta_1" in name_mapping[0]
            and "theta_2" in name_mapping[0]
            and "phi_12" in name_mapping[0]
            and "a_1" in name_mapping[0]
            and "a_2" in name_mapping[0]
            and "iota" in name_mapping[1]
            and "s1_x" in name_mapping[1]
            and "s1_y" in name_mapping[1]
            and "s1_z" in name_mapping[1]
            and "s2_x" in name_mapping[1]
            and "s2_y" in name_mapping[1]
            and "s2_z" in name_mapping[1]
        )

        def named_transform(x):
            iota, s1x, s1y, s1z, s2x, s2y, s2z = spin_to_cartesian_spin(
                x["theta_jn"],
                x["phi_jl"],
                x["theta_1"],
                x["theta_2"],
                x["phi_12"],
                x["a_1"],
                x["a_2"],
                x["M_c"],
                x["q"],
                self.freq_ref,
                x["phase_c"],
            )
            return {
                "iota": iota,
                "s1_x": s1x,
                "s1_y": s1y,
                "s1_z": s1z,
                "s2_x": s2x,
                "s2_y": s2y,
                "s2_z": s2z,
            }

        self.transform_func = named_transform
