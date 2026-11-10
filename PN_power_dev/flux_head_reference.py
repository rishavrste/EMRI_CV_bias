from .base import ODEBase
from ...utils.geodesic import get_fundamental_frequencies, get_separatrix, ELQ_to_pex, _get_separatrix_kernel_inner
from ...utils.globals import get_file_manager
from numba import njit
from ...utils.mappings.kerrecceq import (_kerrecceq_flux_forward_map, 
                                apex_of_uwyz, 
                                apex_of_UWYZ, 
                                z_of_a, 
                                w_of_euz_flux, 
                                e_of_uwz_flux,
                                p_of_u_flux, 
                                u_where_w_is_unity, 
                                u_of_p_flux,
                                DELTAPMIN,
                                EMAX,
                                PMAX_REGIONB,
                                AMAX)
from ...utils.mappings.jacobian import ELdot_to_PEdot_Jacobian
from ...utils.utility import _brentq_jit
from ...utils.exceptions import TrajectoryOffGridException

import h5py

from multispline.spline import BicubicSpline, TricubicSpline
from typing import Union, Optional
import numpy as np
from math import pow, log
import warnings

from .SKequatorialfluxes import (pdot1PA, pdot2PA, edot1PA, edot2PA, OmegaPhi1PA, OmegaPhi2PA, Omegar1PA, Omegar2PA)

PMAX = PMAX_REGIONB
PISCO_MIN = get_separatrix(AMAX, 0, 1)

PISCO_MIN_SCHW = get_separatrix(0, 0, 1) + 1e-5
PMAX_SCHW = 47.6
EMAX_SCHW = 0.755

@njit
def _Edot_PN(e, yPN):
    return (
        (96 + 292 * pow(e, 2) + 37 * pow(e, 4))
        / (15.0 * pow(1 - pow(e, 2), 3.5))
        * pow(yPN, 5)
    )


@njit
def _Ldot_PN(e, yPN):
    return (
        (4 * (8 + 7 * pow(e, 2))) / (5.0 * pow(-1 + pow(e, 2), 2)) * pow(yPN, 7.0 / 2.0)
    )

class SchwarzEccFlux(ODEBase):
    """
    Schwarzschild eccentric flux ODE.

    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)
        # construct the BicubicSpline object from the expected file
        fp = "FluxNewMinusPNScaled_fixed_y_order.dat"

        self.flux_output_convention = "ELQ"

        data = np.loadtxt(get_file_manager().get_file(fp))
        x = np.unique(data[:, 0])
        y = np.unique(data[:, 1])

        self.Edot_interp = BicubicSpline(x, y, data[:, 2].reshape(33, 50).T)
        self.Ldot_interp = BicubicSpline(x, y, data[:, 3].reshape(33, 50).T)

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        return 0.1

    @property
    def supports_ELQ(self):
        return True
    
    def isvalid_x(self, x):
        if np.any(x != 1):
            raise ValueError("Interpolation: x out of bounds. Must be 1.")
        
    def isvalid_e(self, e, e_buffer=[0,0]):
        emax = EMAX_SCHW - e_buffer[1]
        emin = e_buffer[0]
        if np.any(e > emax) or np.any(e < emin):
            raise ValueError(f"Interpolation: e out of bounds. Must be between {emin} and {emax}.")
    
    def isvalid_p(self, p, p_buffer=[0,0]):
        pmax = PMAX_SCHW - p_buffer[1]
        pmin = PISCO_MIN_SCHW + self.separatrix_buffer_dist + p_buffer[0]
        if np.any(p > pmax) or np.any(p < pmin):
            raise ValueError(f"Interpolation: p out of bounds. Must be between {pmin} and {pmax}.")
    
    def isvalid_a(self, a):
        if np.any(a != 0.0):
            raise ValueError(f"Interpolation: a out of bounds. Must be 0.")

    def min_p(self, e, x = 1, a = 0):
        return 6 + 2*e + self.separatrix_buffer_dist

    def max_p(self, e, x = 1, a = 0):
        return PMAX_SCHW + 2.0 * e
    
    def bounds_p(self, e, x = 1, a = 0, p_buffer=[0,0]):
        return [self.min_p(e, x, a) + p_buffer[0], self.max_p(e, x, a) - p_buffer[1]]

    def max_e(self, p, x = 1, a = 0):
        return EMAX_SCHW

    def isvalid_pex(self, p = 20, e = 0, x = 1, a = 0, p_buffer=[0, 0], e_buffer=[0,0]):
        self.isvalid_x(x)
        self.isvalid_e(e, e_buffer=e_buffer)
        self.isvalid_a(a)
        pmin, pmax = self.bounds_p(e, x, a, p_buffer=p_buffer)
        assert (p >= pmin and p <= pmax), f"Interpolation: p out of bounds. Must be between {pmin + p_buffer[0]} and {pmax - p_buffer[1]}."

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 3.817 - np.log((p - 2.0 * e - 2.1))
        dist_e = 0.75 - e

        if dist_p < 0 or dist_e < 0:
            mult = -1
        else:
            mult = 1

        dist = mult * min(abs(dist_p), abs(dist_e))
        return dist

    def interpolate_flux_grids(
        self, p: float, e: float, Omega_phi: float, pLSO: float = None
    ) -> tuple[float]:
        if pLSO is None:
            pLSO = 6.0 + 2.0 * e
        
        if e > 0.755:
            raise ValueError("Interpolation: e out of bounds.")

        y1 = np.log((p - 2.0 * e - 2.1))

        if (
            y1 < 1.3686394258811698 or y1 > 3.817712325956905
        ):  # bounds described in 2104.04582
            raise ValueError(f"Interpolation: p={p} out of bounds.")

        yPN = Omega_phi ** (2 / 3)

        Edot_PN = _Edot_PN(e, yPN)
        Ldot_PN = _Ldot_PN(e, yPN)

        Edot = -(self.Edot_interp(y1, e) * yPN**6 + Edot_PN)
        Ldot = -(self.Ldot_interp(y1, e) * yPN ** (9 / 2) + Ldot_PN)

        return Edot, Ldot

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)
        else:
            p, e, x = y[:3]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)

        Edot, Ldot = self.interpolate_flux_grids(p, e, Omega_phi, pLSO=self.p_sep_cache)

        return [Edot, Ldot, 0.0, Omega_phi, Omega_theta, Omega_r]


@njit 
def _PN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2)**1.5
    Edot = (
        32.0
        / 5.0
        * p ** (-5)
        * oneme2
        * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    )
    Ldot = 32.0 / 5.0 * p ** (-7 / 2) * oneme2 * (1 + 7.0 / 8.0 * e**2)    #here most prob
    return Edot, Ldot

@njit 
def _EdotPN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2)**1.5
    Edot = (
        32.0
        / 5.0
        * p ** (-5)
        * oneme2
        * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)  
    )
    return Edot

@njit 
def _LdotPN_alt(p, e):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    oneme2 = (1 - e**2)**1.5
    Ldot = 32.0 / 5.0 * p ** (-7 / 2) * oneme2 * (1 + 7.0 / 8.0 * e**2)
    return Ldot

@njit
def _emax_w(e, args):
    """
    Function for root-finding the maximum e-value on the domain for a given a, p, x = 1.
    """
    a = args[0]
    p = args[1]
    z = args[2]
    psep = _get_separatrix_kernel_inner(a, e, 1)
    u = u_of_p_flux(p, psep)
    w = w_of_euz_flux(e, u, z)
    return w-1

@njit
def _emax_sep(e, args):
    """
    Function for foot-finding the e-value on the separatrix for a given a, p, x = 1.
    """
    a = args[0]
    p = args[1]
    psep = _get_separatrix_kernel_inner(a, e, 1)
    return p - psep

class KerrEccEqFlux(ODEBase):
    """
    Kerr eccentric equatorial flux ODE.

    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
        downsample: List of two 3-tuples of integers to downsample the flux grid in u, w, z. The first list element
        refers to the inner grid, the second to the outer. Useful for testing error convergence. Defaults to None (no downsampling).
    """

    def __init__(self, *args, use_ELQ: bool = False, downsample=None, flux_output_convention="pex", **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, downsample=downsample, **kwargs)

        self.flux_output_convention = flux_output_convention

        fp = "KerrEccEqFluxData.h5"

        if downsample is None:
            downsample = [(1, 1, 1), (1, 1, 1)]

        downsample_inner = downsample[0]
        downsample_outer = downsample[1]

        fm = get_file_manager()
        file_path = fm.get_file(fp)

        with h5py.File(file_path, "r") as fluxData:
            regionA = fluxData["regionA"]
            u = np.linspace(0, 1, regionA.attrs["NU"])[:: downsample_inner[0]]
            w = np.linspace(0, 1, regionA.attrs["NW"])[:: downsample_inner[1]]
            z = np.linspace(0, 1, regionA.attrs["NZ"])[:: downsample_inner[2]]

            ugrid, wgrid, zgrid = np.asarray(
                np.meshgrid(u, w, z, indexing="ij")
            ).reshape(3, -1)
            agrid, pgrid, egrid, xgrid = apex_of_uwyz(
                ugrid, wgrid, np.ones_like(zgrid), zgrid
            )

            # normalise by PN contribution
            Edot = (
                regionA["Edot"][()][
                    :: downsample_inner[0],
                    :: downsample_inner[1],
                    :: downsample_inner[2],
                ]
            )
            Ldot = (
                regionA["Ldot"][()][
                    :: downsample_inner[0],
                    :: downsample_inner[1],
                    :: downsample_inner[2],
                ]
            )

            if flux_output_convention == "pex":
                # calculate pdot and edot from Edot and Ldot
                Edothere = (Edot).flatten()
                Ldothere = (Ldot).flatten()
                xgrid = np.sign(agrid)
                xgrid[xgrid == 0] = 1
                
                Ldothere = Ldothere * xgrid
                agrid = np.abs(agrid)

                out_pdot_edot = np.asarray([ELdot_to_PEdot_Jacobian(agrid[i], pgrid[i], egrid[i], xgrid[i], Edothere[i], Ldothere[i]) for i in range(Edothere.size)])
                    
                # check whether there are no nans in the output and Edot and Ldot
                if np.isnan(out_pdot_edot).any() or np.isnan(Edot).any() or np.isnan(Ldot).any():
                    raise ValueError("Interpolation: nans in pdot, edot or Edot, Ldot.")
                
                pdot = out_pdot_edot[:, 0].reshape(u.size, w.size, z.size)
                edot = out_pdot_edot[:, 1].reshape(u.size, w.size, z.size)

                risco = get_separatrix(agrid.flatten(), np.zeros_like(agrid.flatten()), xgrid.flatten())
                psep = get_separatrix(agrid.flatten(), egrid.flatten(), xgrid.flatten())
                pdot_pn = _pdot_PN(pgrid.flatten(), egrid.flatten(), risco, psep).reshape(u.size, w.size, z.size) ##this
                edot_pn = _edot_PN(pgrid.flatten(), egrid.flatten(), risco, psep).reshape(u.size, w.size, z.size)

                self.pdot_interp_A = TricubicSpline(u, w, z, pdot / pdot_pn)
                self.edot_interp_A = TricubicSpline(u, w, z, edot / edot_pn)
                
            else:
                EdotPN, LdotPN = _PN_alt(pgrid, egrid)
                EdotPN = EdotPN.reshape(u.size, w.size, z.size)
                LdotPN = LdotPN.reshape(u.size, w.size, z.size)

                self.Edot_interp_A = TricubicSpline(u, w, z, Edot / EdotPN)
                self.Ldot_interp_A = TricubicSpline(u, w, z, Ldot / LdotPN)

            regionB = fluxData["regionB"]
            u = np.linspace(0, 1, regionB.attrs["NU"])[:: downsample_outer[0]]
            w = np.linspace(0, 1, regionB.attrs["NW"])[:: downsample_outer[1]]
            z = np.linspace(0, 1, regionB.attrs["NZ"])[:: downsample_outer[2]]

            ugrid, wgrid, zgrid = np.asarray(
                np.meshgrid(u, w, z, indexing="ij")
            ).reshape(3, -1)
            agrid, pgrid, egrid, xgrid = apex_of_UWYZ(
                ugrid, wgrid, np.ones_like(zgrid), zgrid, True
            )

            # normalise by PN contribution
            Edot = (
                regionB["Edot"][()][
                    :: downsample_outer[0],
                    :: downsample_outer[1],
                    :: downsample_outer[2],
                ]
            )
            Ldot = (
                regionB["Ldot"][()][
                    :: downsample_outer[0],
                    :: downsample_outer[1],
                    :: downsample_outer[2],
                ]
            )

            if self.flux_output_convention == "pex":
                # calculate pdot and edot from Edot and Ldot
                Edothere = (Edot).flatten()
                Ldothere = (Ldot).flatten()
                xgrid = np.sign(agrid)
                xgrid[xgrid == 0] = 1
                
                Ldothere = Ldothere * xgrid
                agrid = np.abs(agrid)

                out_pdot_edot = np.asarray([ELdot_to_PEdot_Jacobian(agrid[i], pgrid[i], egrid[i], xgrid[i], Edothere[i], Ldothere[i]) for i in range(Edothere.size)])
                
                # check whether there are no nans in the output and Edot and Ldot
                if np.isnan(out_pdot_edot).any() or np.isnan(Edot).any() or np.isnan(Ldot).any():
                    raise ValueError("Interpolation: nans in pdot, edot or Edot, Ldot.")
                
                pdot = out_pdot_edot[:, 0].reshape(u.size, w.size, z.size)
                edot = out_pdot_edot[:, 1].reshape(u.size, w.size, z.size)

                risco = get_separatrix(agrid.flatten(), np.zeros_like(agrid.flatten()), xgrid.flatten())
                psep = get_separatrix(agrid.flatten(), egrid.flatten(), xgrid.flatten())
                pdot_pn = _pdot_PN(pgrid.flatten(), egrid.flatten(), risco, psep).reshape(u.size, w.size, z.size)
                edot_pn = _edot_PN(pgrid.flatten(), egrid.flatten(), risco, psep).reshape(u.size, w.size, z.size)

                self.pdot_interp_B = TricubicSpline(u, w, z, pdot / pdot_pn)
                self.edot_interp_B = TricubicSpline(u, w, z, edot / edot_pn)
            else:
                EdotPN, LdotPN = _PN_alt(pgrid, egrid)
                EdotPN = EdotPN.reshape(u.size, w.size, z.size)
                LdotPN = LdotPN.reshape(u.size, w.size, z.size)

                self.Edot_interp_B = TricubicSpline(u, w, z, Edot / EdotPN)
                self.Ldot_interp_B = TricubicSpline(u, w, z, Ldot / LdotPN)

    @property
    def equatorial(self):
        return True

    @property
    def separatrix_buffer_dist(self):
        return 2*DELTAPMIN
    
    @property
    def separatrix_buffer_dist_grid(self):
        return DELTAPMIN

    @property
    def supports_ELQ(self):
        return True
    
    def isvalid_x(self, x):
        if np.any(np.abs(x) != 1):
            raise ValueError("Interpolation: x out of bounds. Must be either 1 or -1.")
        
    def isvalid_e(self, e, e_buffer=[0,0]):
        emax = EMAX - e_buffer[1]
        emin = e_buffer[0]
        if np.any(e > emax) or np.any(e < emin):
            raise ValueError(f"Interpolation: e out of bounds. Must be between {emin} and {emax}.")
    
    def isvalid_p(self, p, p_buffer=[0,0]):
        pmax = PMAX - p_buffer[1]
        pmin = PISCO_MIN + self.separatrix_buffer_dist + p_buffer[0]
        if np.any(p > pmax) or np.any(p < pmin):
            raise ValueError(f"Interpolation: p out of bounds. Must be between {pmin} and {pmax}.")
    
    def isvalid_a(self, a, a_buffer=[0,0]):
        amax = AMAX - a_buffer[1]
        amin = -AMAX + a_buffer[0]
        if np.any(a > amax) or np.any(a < amin):
            raise ValueError(f"Interpolation: a out of bounds. Must be between {amin} and {amax}.")

    def _min_p(self, e, x, a):
        if x == -1:
            a_in = -a
        else:
            a_in = a

        z = z_of_a(a_in)
        p_sep = _get_separatrix_kernel_inner(a, e, x)

        if w_of_euz_flux(e, 0., z) > 1:
            u_min = u_where_w_is_unity(e, z, kind="flux")
        else:
            u_min = 0.

        return max(p_of_u_flux(u_min, p_sep), p_sep + self.separatrix_buffer_dist)

    def _max_p(self, e, x, a):        
        return PMAX
    
    def min_p(self, e = 0, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return self._min_p(e, x, a)
    
    def max_p(self, e = 0, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return self._max_p(e, x, a)

    def _min_e(self, p, x, a):            
        return 0.0

    def _max_e(self, p, x, a):
        if x == -1:
            a_in = -a
        else:
            a_in = a

        p_sep_min_buffer = get_separatrix(a_in, 0, 1) + self.separatrix_buffer_dist
        if p < p_sep_min_buffer:
            raise ValueError(f"Interpolation: p out of bounds. Must be greater than innermost stable circular orbit + buffer = {p_sep_min_buffer}.")
        
        z = z_of_a(a_in)
        e_max_min = e_of_uwz_flux(0, 1, z)
        p_sep_max_min = get_separatrix(a_in, e_max_min, 1)

        p_min = self._min_p(EMAX, x, a)
        if p > p_min:
            emax = EMAX
        else:
            tol = 1e-13
            z = z_of_a(a_in)
            emax = _brentq_jit(_emax_w, 0, EMAX, (a_in, p, z), tol)

            # if you lie below the separatrix, then you are limited by the max e-value on the separatrix
            if get_separatrix(a_in, emax, 1) > p: 
                emax = _brentq_jit(_emax_sep, 0, emax, (a_in, p - self.separatrix_buffer_dist), tol)
        return emax

    def min_e(self, p = 20, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return self._min_e(p, x, a)    

    def max_e(self, p = 20, x = 1, a = 0):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return self._max_e(p, x, a)
    
    def _min_a(self, p, e, x):
        return -AMAX
    
    def _max_a(self, p, e, x):
        return AMAX
    
    def min_a(self, p = 20, e = 0, x = 1):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return self._min_a(p,e,x)
    
    def max_a(self, p = 20, e = 0, x = 1):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return self._max_a(p,e,x)
    
    def bounds_a(self, p = 20, e = 0, x = 1, a_buffer = [0,0]):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_e(e)
        return [self._min_a(p, e, x) + a_buffer[0], self._max_a(p, e, x) - a_buffer[1]]
    
    def bounds_p(self, e = 0, x = 1, a = 0, p_buffer = [0,0]):
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return [self._min_p(e, x, a) + p_buffer[0], self._max_p(e, x, a) - p_buffer[1]]
    
    def bounds_e(self, p = 20, x = 1, a = 0, e_buffer = [0,0]):
        self.isvalid_x(x)
        self.isvalid_p(p)
        self.isvalid_a(a)
        return [self._min_e(p, x, a) + e_buffer[0], self._max_e(p, x, a) - e_buffer[1]]
    
    def isvalid_pex(self, p = 20, e = 0, x = 1, a = 0, p_buffer = [0,0], e_buffer=[0,0], a_buffer=[0,0]):
        self.isvalid_x(x)
        self.isvalid_e(e, e_buffer=e_buffer)
        self.isvalid_a(a, a_buffer=a_buffer)
        pmin, pmax = self.bounds_p(e, x, a, p_buffer=p_buffer)
        assert (p >= pmin and p <= pmax), f"Interpolation: p {p} out of bounds. Must be between {pmin} and {pmax}."

    def interpolate_flux_grids(self, p: float, e: float, x: float = 1, a: float = 0, pLSO: Optional[float] = None) -> tuple[float]:
        if pLSO is None:
            pLSO = get_separatrix(a, e, x)

        edge_buffer = -1e-8

        # handle xI = -1 case
        if x == -1:
            a_in = -a
        else:
            a_in = a

        u, w, _, z, in_region_A = _kerrecceq_flux_forward_map(
            a_in, p, e, 1.0, pLSO
        )
        
        if u < edge_buffer or u > 1 - edge_buffer or np.isnan(u):
            raise ValueError("Interpolation: p out of bounds.")
        if w < edge_buffer:
            raise TrajectoryOffGridException("Interpolation: e out of bounds.")
        if w > 1 - edge_buffer:
            if self.integrate_backwards:
                raise ValueError("Interpolation: e out of bounds.")
            else:
                raise TrajectoryOffGridException("Interpolation: e out of bounds.")

        if z < edge_buffer or z > 1 - edge_buffer:
            raise TrajectoryOffGridException("Interpolation: a out of bounds.")

        if self.flux_output_convention == "ELQ": ##this
            EdotPN, LdotPN = _PN_alt(p, e)
            if in_region_A:
                Edot = -self.Edot_interp_A(u, w, z) * EdotPN
                Ldot = -self.Ldot_interp_A(u, w, z) * LdotPN
            else:
                Edot = -self.Edot_interp_B(u, w, z) * EdotPN
                Ldot = -self.Ldot_interp_B(u, w, z) * LdotPN

            if a_in < 0:
                Ldot *= -1

            return Edot, Ldot

        else:
            risco = get_separatrix(a_in, 0.0, 1.0)
            p_sep = pLSO
            pdotPN = _pdot_PN(p, e, risco, p_sep)
            edotPN = _edot_PN(p, e, risco, p_sep)
            if in_region_A:
                pdot = -self.pdot_interp_A(u, w, z) * pdotPN
                edot = -self.edot_interp_A(u, w, z) * edotPN
            else:
                pdot = -self.pdot_interp_B(u, w, z) * pdotPN
                edot = -self.edot_interp_B(u, w, z) * edotPN

            return pdot, edot

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)
        else:
            p, e, x = y[:3]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)

        Edot, Ldot = self.interpolate_flux_grids(p, e, x, a=self.a, pLSO=self.p_sep_cache)

        return [Edot, Ldot, 0.0, Omega_phi, Omega_theta, Omega_r]

class SuperKludgeFlux(KerrEccEqFlux):
    """
    SuperKludgeFlux as a modification of the Kerr eccentric equatorial flux ODE.
    Based on transformed gauge such that eccentric -> circular limit now works well.

    Additional parameters (in this order):
        chi2 (float) : dimensionless spin of the secondary.
        evolve_1PA (bool) : whether to include 1PA corrections.
        evolve_primary (bool) : whether to evolve MBH mass M and spin a/chi1.
        evolve_2PA (bool) : whether to include 2PA corrections.
        deviation_included (bool) : master switch for all deviation parameters below.
                                    If False, every deviation coefficient is set to zero.

    Deviation parameters (all optional, all default to 0.0, i.e. GR).
    The two families are independent and may be used together or separately:

        2.5PN-type deviation (additive correction to pdot, edot):
            C_p (float) : index 5. Coefficient of the pdot deviation term.
            C_e (float) : index 6. Coefficient of the edot deviation term.

        Multiplicative deviation on the adiabatic fluxes:
            del_0_p (float) : index 7. Scales the energy flux, Edot -> (1 + eta * del_0_p) Edot.
            del_0_e (float) : index 8. Scales the angular momentum flux, Ldot -> (1 + eta * del_0_e) Ldot.

    All additional arguments must be numeric: they are passed through ``np.asarray`` upstream,
    so a single string entry would promote the whole array to a string dtype.
    """


    def _get_deviation(self, additional_args, index: int, name: str) -> float:
        """
        Read an optional deviation coefficient from ``additional_args``.

        Missing entries default to 0.0, which recovers GR for that coefficient. An entry that is
        present but not numeric is a user error, so it warns before falling back to 0.0.
        """
        try:
            value = additional_args[index]
        except (IndexError, TypeError):
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            warnings.warn(f"Deviation parameter {name} is not numeric ({value!r}). Defaulting to zero.")
            return 0.0

    def add_fixed_parameters(self, m1: float, m2: float, a: float, additional_args = None):

        #print("additional args: ", additional_args)

        #this is where we initialize additional args like chi2, massratio, flags for 1PA, 2PA, primary evolution.
                
        self.massratio = m1 * m2 / (m1 + m2) ** 2
        self.m1 = m1
        
        self.a = a
        self.chi2 = additional_args[0] #secondary spin (dimless)
        try:
            self.evolve_1PA = bool(additional_args[1]) #whether to include 1PA corrections.
        except IndexError:
            self.evolve_1PA = False #defaults to True
            
        try:
            self.evolve_primary = bool(additional_args[2]) #whether to evolve \delta~M, \delta~a. If False, just set delta_m1, delta_a = 0.0 throughout evolution.
            if self.evolve_primary:
                warnings.warn("Flux at horizon for primary evolution are PN-approximated and may be incorrect in the strong field.")
        except IndexError:
            self.evolve_primary = False #do not include primary evolution.
        
        try:
            self.evolve_2PA = bool(additional_args[3]) #whether to include 2PA corrections
        except IndexError:
            self.evolve_2PA = False #defaults to True

        try:
            self.deviation_included = bool(additional_args[4]) #whether to add deviation
       
        except IndexError:
            self.deviation_included = False #defaults to False

        #deviation coefficients. Each family lives in its own slots, so they can be used
        #together or separately. Anything not supplied stays at zero, i.e. GR.
        deviation_slots = {
            "C_p": 5,        #0PN-1PN-type additive pdot deviation
            "C_e": 6,        #0PN-1PN-type additive edot deviation
            "n_p": 7,    #power lawa rescaling
            "n_e": 8,    #power law rescaling
        }

        for name, index in deviation_slots.items():
            if self.deviation_included:
                setattr(self, name, self._get_deviation(additional_args, index, name))
            else:
                setattr(self, name, 0.0)

        #print("evolve_1PA: ", self.evolve_1PA, "evolve_primary: ", self.evolve_primary, "evolve_2PA: ", self.evolve_2PA,"Deviation_Include",self.deviation_included,self.C_p,self.chi2)
        
        if additional_args is None:
            self.num_add_args = 0
        else:
            self.num_add_args = len(additional_args)

    @property
    def separatrix_buffer_dist(self):
        return 0.05 #large-ish buffer for final p to separatrix so that the trajectories are stable.

    @property
    def nparams(self):
        """
        An integer describing the number of parameters this ODE will integrate.
        Defaults to 6 (three orbital elements, three orbital phases).
        """
        return 8 #[p, e, x, Phi_phi0, Phi_r, Phi_theta, delta_m1, delta_a]

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:

        a_at_t = self.a + y[-1] #evolving spin
        
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(a_at_t, E, L, Q)
        else:
            p, e, x = y[:3]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(a_at_t, p, e, x)

        Edot, Ldot = self.interpolate_flux_grids(p, e, x, a=a_at_t, pLSO=self.p_sep_cache)

        return [Edot, Ldot, 0.0, Omega_phi, Omega_theta, Omega_r, 0.0, 0.0] #we will add delta_m1_dot, delta_a_dot in modify_rhs

    def modify_rhs( self, ydot: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        This function allows the user to modify the right-hand side of the ODE after any required Jacobian transforms
        have been applied. Note: modification is in place.
        !!! m1 is the MBH mass in solar masses. TM is the total mass (m1 + m2) in solar masses.!!!
        """

        pdot, edot, _, Omega_phi, Omega_theta, Omega_r = ydot[:6]
        p, e = y[:2]

        delta_m1 = y[-2]
        delta_a = y[-1]
        delta_m1_dot = ydot[-2]
        delta_a_dot = ydot[-2]
        
        M_at_t = self.m1 + delta_m1 #this is how delta_m1 is defined
        a_at_t = self.a + delta_a #this is how delta_a is defined

        #evolution of MBH mass and spin
        if self.evolve_primary:

            return NotImplementedError
            
            """
            #we will have Edot, Ldot at horizon as a function of p, e, x.
            #these are PN approximations at 1PA.
            EdotH = self.m1 * self.massratio * dEdtH1PA(p, e, 1.0, a_at_t) #Energy flux at horizon, 1PA contribution, scaled by the MBH mass.
            LdotH = self.m1 ** 2 * self.massratio * dLdtH1PA(p, e, 1.0, a_at_t) #Angular momentum flux at horizon, 1PA contribution, scaled by the MBH mass ** 2.

            #calculate delta_m1_dot, delta_a_dot as functions of EdotH, LdotH
            S1dot = - LdotH #should be positive.
            
            #print(S1dot)
            
            delta_m1_dot = - EdotH #same as M_dot
            delta_a_dot = - (2 * (a_at_t) * delta_m1_dot)/(M_at_t) + S1dot/(M_at_t)**2 #same as a_dot. S1dot should be positive.

            #print(EdotH, LdotH)#delta_m1, delta_m1_dot, delta_a, delta_a_dot)
                        
            #update ydot
            ydot[-2:] = [delta_m1_dot, delta_a_dot]
            
            #add contribution to pdot, edot due to delta_m1_dot, delta_a_dot
            pdot += ((dpdE(a_at_t, p, e, 1.0)*dEdChi1(a_at_t, p, e, 1.0) + dpdL(a_at_t, p, e, 1.0)*dLdChi1(a_at_t, p, e, 1.0))*delta_a_dot +
                    (dpdE(a_at_t, p, e, 1.0)*dEdm1(a_at_t, p, e, 1.0, M_at_t) + dpdL(a_at_t, p, e, 1.0)*dLdm1(a_at_t, p, e, 1.0, M_at_t))*delta_m1_dot)

            edot += ((dedE(a_at_t, p, e, 1.0)*dEdChi1(a_at_t, p, e, 1.0) + dedL(a_at_t, p, e, 1.0)*dLdChi1(a_at_t, p, e, 1.0))*delta_a_dot +
                    (dedE(a_at_t, p, e, 1.0)*dEdm1(a_at_t, p, e, 1.0, M_at_t) + dedL(a_at_t, p, e, 1.0)*dLdm1(a_at_t, p, e, 1.0, M_at_t))*delta_m1_dot)
            """
            
        if self.deviation_included:

            #0PN-1PN-type additive corrections. Zero coefficients recover GR.
            #print("Adding deviation with C_p: ", self.C_p, "C_e: ", self.C_e)
            pdot +=(self.massratio * self.C_p * ((1-e**2)**1.5) * ((8 + 7 * e **2 )/p ** 3.5))*((p/10)**(self.n_p)) #added deviation
            edot +=(self.massratio * e * self.C_e * ((1-e**2)**1.5) * ((304 + 121 * e **2 )/p ** 4.5))*((p/10)**(self.n_e)) #added deviation

        if self.evolve_1PA:

            #adding 1PA corrections:
            pdot1PAval = self.massratio * pdot1PA(a_at_t, p, e, self.chi2) #adiabatic pdot, edot are scaled by the massratio. So we lose one factor of massratio here.
            pdot +=pdot1PAval    

            edot1PAval = self.massratio * edot1PA(a_at_t, p, e, self.chi2)
            edot +=edot1PAval        #added deviation

            Omega_phi_1PAval = self.massratio * OmegaPhi1PA(a_at_t, p, e, self.chi2) #adiabatic Omega_phi, Omega_r NOT scaled by the massratio. So we keep the factor of massratio here.
            Omega_phi += Omega_phi_1PAval

            Omega_r_1PAval = self.massratio * Omegar1PA(a_at_t, p, e, self.chi2)
            Omega_r += Omega_r_1PAval


        if self.evolve_2PA:

            #adding 2PA corrections:
            pdot2PAval = self.massratio**2 * pdot2PA(a_at_t, p, e, self.chi2)
            pdot +=pdot2PAval     #added deviation
            
            edot2PAval = self.massratio**2 * edot2PA(a_at_t, p, e, self.chi2)
            edot +=edot2PAval  #added deviation

            Omega_phi_2PAval = self.massratio**2 * OmegaPhi2PA(a_at_t, p, e, self.chi2)
            Omega_phi += Omega_phi_2PAval

            Omega_r_2PAval = self.massratio**2 * Omegar2PA(a_at_t, p, e, self.chi2)
            Omega_r += Omega_r_2PAval
        
        ydot[0] = pdot #pdot
        ydot[1] = edot #edot
        ydot[3] = Omega_phi #Omega_phi
        ydot[5] = Omega_r #Omega_r

    def __call__(
        self,
        y: Union[list, np.ndarray],
        out: Optional[np.ndarray] = None,
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        in_bounds = self.cache_values_and_check_bounds(y)

        if out is None:
            out = np.zeros(8) #8 params in SuperKludge (3 orbital params, 3 phases, 2 evolution of primary)
            
        if in_bounds:
            out[:] = self.evaluate_rhs(y, **kwargs)
        else:
            out *= np.nan

        self.modify_rhs_before_Jacobian(out, y, **kwargs)

        if self.apply_Jacobian_bool:  # implicitly this means that y contains (p, e, x)
            out[:2] = ELdot_to_PEdot_Jacobian(self.a + y[-1], *y[:3], *out[:2])

        self.modify_rhs(out, y, **kwargs)

        if self.integrate_backwards:
            out *= -1.

        return out

@njit
def _pdot_PN(p, e, risco, p_sep):
    return (8.0 * (1.0 - (e * e))** 1.5 * (8.0 + 7.0 * (e * e))) / (
        5.0 * p * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )

@njit
def _edot_PN(p, e, risco, p_sep):
    return (((1.0 - (e * e)) ** 1.5) * (304.0 + 121.0 * (e * e))) / (
        15.0
        * (p * p)
        * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )


@njit(fastmath=True)
def _p_to_u(p, p_sep):
    return log((p - p_sep + 4.0 - 0.05) / 4)


class KerrEccEqFluxLegacy(ODEBase):
    """
    Kerr eccentric equatorial flux ODE.

    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)
        self.files = [
            "KerrEqEcc_x0.dat",
            "KerrEqEcc_x1.dat",
            "KerrEqEcc_x2.dat",
            "KerrEqEcc_pdot.dat",
            "KerrEqEcc_edot.dat",
        ]
        fm = get_file_manager()
        fm.prefetch_files_by_list(self.files)

        x = np.loadtxt(fm.get_file(self.files[0]))
        y = np.loadtxt(fm.get_file(self.files[1]))
        z = np.loadtxt(fm.get_file(self.files[2]))

        pdot = np.loadtxt(fm.get_file(self.files[3])).reshape(x.size, y.size, z.size)
        edot = np.loadtxt(fm.get_file(self.files[4])).reshape(x.size, y.size, z.size)

        self.pdot_interp = TricubicSpline(x, y, z, np.log(-pdot))
        self.edot_interp = TricubicSpline(x, y, z, edot)

    @property
    def equatorial(self):
        return True

    @property
    def separatrix_buffer_dist(self):
        return 0.05

    @property
    def supports_ELQ(self):
        return False

    @property
    def flux_output_convention(self):
        return "pex"

    def interpolate_flux_grids(self, p: float, e: float, x: float) -> tuple[float]:
        risco = get_separatrix(self.a, 0.0, x)
        u = _p_to_u(p, self.p_sep_cache)
        w = e**0.5
        a_sign = self.a * x

        pdot = -np.exp(self.pdot_interp(a_sign, w, u)) * _pdot_PN(
            p, e, risco, self.p_sep_cache
        )
        edot = self.edot_interp(a_sign, w, u) * _edot_PN(p, e, risco, self.p_sep_cache)

        return pdot, edot

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            raise NotImplementedError
        else:
            p, e, x = y[:3]

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)

        pdot, edot = self.interpolate_flux_grids(p, e, x)

        return [pdot, edot, 0.0, Omega_phi, Omega_theta, Omega_r]
