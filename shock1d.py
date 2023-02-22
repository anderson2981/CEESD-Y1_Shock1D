"""Shock tube problem."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
import sys
import yaml
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
import math
from dataclasses import dataclass, fields
from pytools.obj_array import make_obj_array
from functools import partial
from mirgecom.discretization import create_discretization_collection

from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import VolumeDomainTag, DOFDesc
import grudge.op as op
from grudge.dof_desc import DD_VOLUME_ALL
from grudge.trace_pair import inter_volume_trace_pairs
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage
)

from mirgecom.artificial_viscosity import smoothness_indicator
from mirgecom.simutil import (
    check_step,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    force_evaluation
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)
from mirgecom.inviscid import (inviscid_facial_flux_rusanov,
                               inviscid_facial_flux_hll)
from grudge.shortcuts import compiled_lsrk45_step

from mirgecom.fluid import make_conserved
from mirgecom.limiter import bound_preserving_limiter
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    IsothermalWallBoundary,
    SymmetryBoundary,
    AdiabaticSlipBoundary,
    AdiabaticNoslipWallBoundary,
    DummyBoundary
)
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary
)
from mirgecom.initializers import PlanarDiscontinuity
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.transport import (SimpleTransport,
                                PowerLawTransport,
                                ArtificialViscosityTransportDiv)
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    coupled_grad_t_operator,
    coupled_ns_heat_operator
)
from mirgecom.navierstokes import grad_cv_operator


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class _OxCommTag:
    pass


class _FluidOxDiffCommTag:
    pass


class _WallOxDiffCommTag:
    pass


def smooth_step(actx, x, epsilon=1e-12):
    return (
        actx.np.greater(x, 0) * actx.np.less(x, 1) * (1 - actx.np.cos(np.pi*x))/2
        + actx.np.greater(x, 1))


from mirgecom.viscous import viscous_facial_flux_central
from grudge.trace_pair import TracePair
from grudge.dof_desc import as_dofdesc


class PlanarDiscontinuityMulti:
    r"""Solution initializer for flow with a discontinuity.

    This initializer creates a physics-consistent flow solution
    given an initial thermal state (pressure, temperature) and an EOS.

    The solution varies across a planar interface defined by a tanh function
    located at disc_location for pressure, temperature, velocity, and mass fraction

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=3, normal_dir, disc_location, disc_location_species,
            nspecies=0,
            temperature_left, temperature_right,
            pressure_left, pressure_right,
            velocity_left=None, velocity_right=None,
            velocity_cross=None,
            species_mass_left=None, species_mass_right=None,
            convective_velocity=None, sigma=0.5,
            temp_sigma=0., vel_sigma=0., temp_wall=300.
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        normal_dir: numpy.ndarray
            specifies the direction (plane) the discontinuity is applied in
        disc_location: numpy.ndarray or Callable
            fixed location of discontinuity or optionally a function that
            returns the time-dependent location.
        disc_location_species: numpy.ndarray or Callable
            fixed location of the species discontinuity
        nspecies: int
            specifies the number of mixture species
        pressure_left: float
            pressure to the left of the discontinuity
        temperature_left: float
            temperature to the left of the discontinuity
        velocity_left: numpy.ndarray
            velocity (vector) to the left of the discontinuity
        species_mass_left: numpy.ndarray
            species mass fractions to the left of the discontinuity
        pressure_right: float
            pressure to the right of the discontinuity
        temperature_right: float
            temperaure to the right of the discontinuity
        velocity_right: numpy.ndarray
            velocity (vector) to the right of the discontinuity
        species_mass_right: numpy.ndarray
            species mass fractions to the right of the discontinuity
        sigma: float
           sharpness parameter
        velocity_cross: numpy.ndarray
            velocity (vector) tangent to the shock
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        """
        if velocity_left is None:
            velocity_left = np.zeros(shape=(dim,))
        if velocity_right is None:
            velocity_right = np.zeros(shape=(dim,))
        if velocity_cross is None:
            velocity_cross = np.zeros(shape=(dim,))

        if species_mass_left is None:
            species_mass_left = np.zeros(shape=(nspecies,))
        if species_mass_right is None:
            species_mass_right = np.zeros(shape=(nspecies,))

        self._nspecies = nspecies
        self._dim = dim
        self._disc_location = disc_location
        self._disc_location_species = disc_location_species
        self._sigma = sigma
        self._ul = velocity_left
        self._ur = velocity_right
        self._ut = velocity_cross
        self._uc = convective_velocity
        self._pl = pressure_left
        self._pr = pressure_right
        self._tl = temperature_left
        self._tr = temperature_right
        self._yl = species_mass_left
        self._yr = species_mass_right
        self._normal = normal_dir
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._temp_wall = temp_wall

    def __call__(self, x_vec, eos, *, time=0.0):
        """Create the mixture state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        xpos = x_vec[0]
        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]

        actx = xpos.array_context
        #if isinstance(self._disc_location, Number):
        if callable(self._disc_location):
            x0 = self._disc_location(time)
        else:
            x0 = self._disc_location

        if callable(self._disc_location_species):
            x0_species = self._disc_location(time)
        else:
            x0_species = self._disc_location_species

        # get the species mass fractions first
        dist = np.dot(x0_species - x_vec, self._normal)
        xtanh = 1.0/self._sigma*dist
        weight = 0.5*(1.0 - actx.np.tanh(xtanh))
        y = self._yl + (self._yr - self._yl)*weight

        # now solve for T, P, velocity
        dist = np.dot(x0 - x_vec, self._normal)
        xtanh = 1.0/self._sigma*dist
        weight = 0.5*(1.0 - actx.np.tanh(xtanh))
        pressure = self._pl + (self._pr - self._pl)*weight
        temperature = self._tl + (self._tr - self._tl)*weight
        velocity = self._ul + (self._ur - self._ul)*weight + self._ut

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        y_top = 0.01
        y_bottom = -0.01
        if self._temp_sigma > 0:
            sigma = self._temp_sigma
            wall_temperature = self._temp_wall
            smoothing_top = smooth_step(actx, -sigma*(ypos - y_top))
            smoothing_bottom = smooth_step(actx, sigma*(ypos - y_bottom))
            temperature = (wall_temperature +
                           (temperature - wall_temperature)*smoothing_top*smoothing_bottom)

        # modify the velocity in the near wall region to match the
        # noslip boundaries
        sigma = self._vel_sigma
        smoothing_top = smooth_step(actx, -sigma*(ypos - y_top))
        smoothing_bottom = smooth_step(actx, sigma*(ypos - y_bottom))
        velocity[0] = velocity[0]*smoothing_top*smoothing_bottom

        if self._nspecies:
            mass = eos.get_density(pressure, temperature,
                                   species_mass_fractions=y)
        else:
            mass = pressure/temperature/eos.gas_const()

        specmass = mass * y
        mom = mass * velocity
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)

        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


def _get_normal_axes(actx, seed_vector):
    vec_dim, = seed_vector.shape

    vec_mag = actx.np.sqrt(np.dot(seed_vector, seed_vector))
    seed_vector = seed_vector / vec_mag

    if vec_dim == 1:
        return seed_vector,

    if vec_dim == 2:
        vector_2 = 0*seed_vector
        vector_2[0] = -1.*seed_vector[1]
        vector_2[1] = 1.*seed_vector[0]
        return seed_vector, vector_2

    if vec_dim == 3:
        x_comp = seed_vector[0]
        y_comp = seed_vector[1]
        z_comp = seed_vector[2]
        zsign = z_comp / actx.np.abs(z_comp)

        a = vec_mag * zsign
        b = z_comp + a

        vector_2 = 0*seed_vector
        vector_2[0] = a*b - x_comp*x_comp
        vector_2[1] = -x_comp*y_comp
        vector_2[2] = -x_comp*b
        vec_mag2 = actx.np.sqrt(np.dot(vector_2, vector_2))
        vector_2 = vector_2 / vec_mag2
        x_comp_2 = vector_2[0]
        y_comp_2 = vector_2[1]
        z_comp_2 = vector_2[2]

        vector_3 = 0*vector_2
        vector_3[0] = y_comp*z_comp_2 - y_comp_2*z_comp
        vector_3[1] = x_comp_2*z_comp - x_comp*z_comp_2
        vector_3[2] = x_comp*y_comp_2 - y_comp*x_comp_2

    return seed_vector, vector_2, vector_3


class AdiabaticSlipWallBoundary2(PrescribedFluidBoundary):
    r"""Adiabatic slip viscous wall boundary.

    This class implements an adiabatic slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: inviscid_wall_flux
    .. automethod:: viscous_wall_flux
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: adiabatic_wall_state
    """

    def __init__(self):
        """Initialize the boundary condition object."""
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_slip_wall_state,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_gradient_temperature_func=self.grad_temperature_bc,
            boundary_gradient_cv_func=self.grad_cv_bc
        )

        print(f"{self._bnd_state_func=}")
        print(f"{self._temperature_grad_flux_func=}")
        print(f"{self._inviscid_flux_func=}")
        print(f"{self._bnd_temperature_func=}")
        print(f"{self._grad_num_flux_func=}")
        print(f"{self._cv_gradient_flux_func=}")
        print(f"{self._viscous_flux_func=}")
        print(f"{self._bnd_grad_cv_func=}")
        print(f"{self._bnd_grad_temperature_func=}")
        print(f"{self._av_num_flux_func=}")
        print(f"{self._bnd_grad_av_func=}")

    def adiabatic_slip_wall_state(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with zero normal-component velocity
           and the respective internal energy for an adiabatic wall."""

        actx = state_minus.array_context

        # Grab a unit normal to the boundary
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        cv_minus = state_minus.cv
        # The wall-normal component of momentum
        mom_norm = np.dot(cv_minus.momentum, nhat)*nhat

        # set the normal momentum to 0
        mom_plus = cv_minus.momentum - mom_norm


        # subtract off the total energy lost from modifying the velocity
        # this keeps the temperature on the plus side equal to the minus
        internal_energy_plus = (state_minus.energy_density
            - 0.5*np.dot(cv_minus.momentum, cv_minus.momentum)/cv_minus.mass)
        total_energy_plus = (internal_energy_plus
                             + 0.5*np.dot(mom_plus, mom_plus)/cv_minus.mass)

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density, energy=total_energy_plus,
            momentum=mom_plus, species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def inviscid_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                           numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        """
        Compute the flux such that there will be vanishing
        flux through the boundary, preserving mass, momentum (magnitude) and
        energy.
        rho_plus = rho_minus
        v_plus = v_minus - 2 * (v_minus . n_hat) * n_hat
        mom_plus = rho_plus * v_plus
        E_plus = E_minus
        """
        dd_bdry = as_dofdesc(dd_bdry)

        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        ext_mom = (state_minus.momentum_density
                   - 2.0*np.dot(state_minus.momentum_density, normal)*normal)

        wall_cv = make_conserved(dim=state_minus.dim,
                                 mass=state_minus.mass_density,
                                 momentum=ext_mom,
                                 energy=state_minus.energy_density,
                                 species_mass=state_minus.species_mass_density)
        wall_state = make_fluid_state(cv=wall_cv, gas_model=gas_model,
                                      temperature_seed=state_minus.temperature,
                                      smoothness=state_minus.smoothness)
        #print(f"{state_minus.pressure=}")
        #print(f"{wall_state.pressure=}")
        state_pair = TracePair(dd_bdry, interior=state_minus, exterior=wall_state)

        return numerical_flux_func(state_pair, gas_model, normal)

    def grad_temperature_bc(self, grad_t_minus, normal, **kwargs):
        """The temperature gradient on the plus state, no normal component."""
        return (grad_t_minus
                - np.dot(grad_t_minus, normal)*normal)

    def grad_cv_bc(self, state_minus, state_plus, grad_cv_minus, normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""

        dim = state_minus.dim
        actx = state_minus.array_context

        grad_species_mass_plus = 1.*grad_cv_minus.species_mass
        if state_minus.nspecies > 0:
            from mirgecom.fluid import species_mass_fraction_gradient
            grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                          grad_cv_minus)
            grad_y_plus = grad_y_minus - np.outer(grad_y_minus@normal, normal)
            grad_species_mass_plus = 0.*grad_y_plus

            for i in range(state_minus.nspecies):
                grad_species_mass_plus[i] = \
                    (state_minus.mass_density*grad_y_plus[i]
                     + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        # normal velocity on the surface is zero,
        vel_plus = state_plus.velocity
        #print(f"{state_minus.velocity=}")
        #print(f"{vel_plus=}")

        principle_axis = _get_normal_axes(actx, normal)
        dim, = normal.shape
        comps = []

        for d in range(dim):
            axis = principle_axis[d]
            for i in range(dim):
                comps.append(axis[i])

        comps = make_obj_array(comps)
        rotation_matrix = comps.reshape(dim, dim)

        # subtract the normal-component of velocity gradient to get the tangential part
        from mirgecom.fluid import velocity_gradient
        grad_v_minus = velocity_gradient(state_minus.cv, grad_cv_minus)

        # rotate the gradient tensor into the normal direction
        grad_v_minus_normal = rotation_matrix@grad_v_minus@rotation_matrix.T

        # set the normal component of the tangential velocity to 0
        #print(f"{grad_v_minus_normal.shape=}")
        for i in range(dim-1):
            grad_v_minus_normal[i+1][0] = 0.*grad_v_minus_normal[0][0]
            grad_v_minus_normal[0][i+1] = 0.*grad_v_minus_normal[0][0]

        grad_v_minus_normal[1][0] = -grad_v_minus_normal[0][1]

        # get the gradient on the plus side in the original coordiate space
        grad_v_plus = rotation_matrix.T@grad_v_minus_normal@rotation_matrix

        # construct grad(mom)
        grad_mom_plus = (state_minus.mass_density*grad_v_plus
                         + np.outer(vel_plus, grad_cv_minus.mass))

        # modify grad(rhoE)
        # MJA, not needed? gradients are needed for the viscous fluxes and
        # energy does not appear there

        return make_conserved(grad_cv_minus.dim,
                              mass=grad_cv_minus.mass,
                              energy=grad_cv_minus.energy,
                              momentum=grad_mom_plus,
                              species_mass=grad_species_mass_plus)

    def viscous_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                          grad_cv_minus, grad_t_minus,
                          numerical_flux_func=viscous_facial_flux_central,
                          **kwargs):
        """Return the boundary flux for the divergence of the viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)

        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        state_wall = self.adiabatic_slip_wall_state(dcoll=dcoll, dd_bdry=dd_bdry,
                                                    gas_model=gas_model,
                                                    state_minus=state_minus,
                                                    **kwargs)

        grad_cv_wall = self.grad_cv_bc(state_minus=state_minus,
                                       state_plus=state_wall,
                                       grad_cv_minus=grad_cv_minus,
                                       normal=normal, **kwargs)

        grad_t_wall = self.grad_temperature_bc(
            grad_t_minus=grad_t_minus,
            normal=normal, **kwargs)

        f_ext = viscous_flux(state_wall, grad_cv_wall, grad_t_wall)
        return f_ext@normal


class AdiabaticSlipWallBoundary(PrescribedFluidBoundary):
    r"""Adiabatic slip viscous wall boundary.

    This class implements an adiabatic slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: inviscid_wall_flux
    .. automethod:: viscous_wall_flux
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: adiabatic_wall_state
    """

    def __init__(self):
        """Initialize the boundary condition object."""
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_slip_wall_state,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_gradient_temperature_func=self.grad_temperature_bc,
            boundary_gradient_cv_func=self.grad_cv_bc
        )

        print(f"{self._bnd_state_func=}")
        print(f"{self._temperature_grad_flux_func=}")
        print(f"{self._inviscid_flux_func=}")
        print(f"{self._bnd_temperature_func=}")
        print(f"{self._grad_num_flux_func=}")
        print(f"{self._cv_gradient_flux_func=}")
        print(f"{self._viscous_flux_func=}")
        print(f"{self._bnd_grad_cv_func=}")
        print(f"{self._bnd_grad_temperature_func=}")
        print(f"{self._av_num_flux_func=}")
        print(f"{self._bnd_grad_av_func=}")

    def adiabatic_slip_wall_state(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with zero normal-component velocity
           and the respective internal energy for an adiabatic wall."""

        actx = state_minus.array_context

        # Grab a unit normal to the boundary
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        cv_minus = state_minus.cv
        # The wall-normal component of momentum
        mom_norm = np.dot(cv_minus.momentum, nhat)*nhat

        # set the normal momentum to 0
        mom_plus = cv_minus.momentum - mom_norm


        # subtract off the total energy lost from modifying the velocity
        # this keeps the temperature on the plus side equal to the minus
        internal_energy_plus = (state_minus.energy_density
            - 0.5*np.dot(cv_minus.momentum, cv_minus.momentum)/cv_minus.mass)
        total_energy_plus = (internal_energy_plus
                             + 0.5*np.dot(mom_plus, mom_plus)/cv_minus.mass)

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density, energy=total_energy_plus,
            momentum=mom_plus, species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def inviscid_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                           numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        """
        Compute the flux such that there will be vanishing
        flux through the boundary, preserving mass, momentum (magnitude) and
        energy.
        rho_plus = rho_minus
        v_plus = v_minus - 2 * (v_minus . n_hat) * n_hat
        mom_plus = rho_plus * v_plus
        E_plus = E_minus
        """
        dd_bdry = as_dofdesc(dd_bdry)

        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        ext_mom = (state_minus.momentum_density
                   - 2.0*np.dot(state_minus.momentum_density, normal)*normal)

        wall_cv = make_conserved(dim=state_minus.dim,
                                 mass=state_minus.mass_density,
                                 momentum=ext_mom,
                                 energy=state_minus.energy_density,
                                 species_mass=state_minus.species_mass_density)
        wall_state = make_fluid_state(cv=wall_cv, gas_model=gas_model,
                                      temperature_seed=state_minus.temperature,
                                      smoothness=state_minus.smoothness)
        state_pair = TracePair(dd_bdry, interior=state_minus, exterior=wall_state)

        return numerical_flux_func(state_pair, gas_model, normal)

    def grad_temperature_bc(self, grad_t_minus, normal, **kwargs):
        #"""The temperature gradient on the plus state, no normal component."""
        #return (grad_t_minus
                   #- np.dot(grad_t_minus, normal)*normal)
        """The temperature gradient on the plus state, opposite normal component."""
        return (grad_t_minus
                   - 2.*np.dot(grad_t_minus, normal)*normal)

    def grad_cv_bc(self, state_minus, state_plus, grad_cv_minus, normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""

        dim = state_minus.dim
        actx = state_minus.array_context

        grad_species_mass_plus = 1.*grad_cv_minus.species_mass
        if state_minus.nspecies > 0:
            from mirgecom.fluid import species_mass_fraction_gradient
            grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                          grad_cv_minus)
            grad_y_plus = grad_y_minus - np.outer(grad_y_minus@normal, normal)
            grad_species_mass_plus = 0.*grad_y_plus

            for i in range(state_minus.nspecies):
                grad_species_mass_plus[i] = \
                    (state_minus.mass_density*grad_y_plus[i]
                     + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        # normal velocity on the surface is zero,
        vel_plus = state_plus.velocity
        #print(f"{state_minus.velocity=}")
        #print(f"{vel_plus=}")

        principle_axis = _get_normal_axes(actx, normal)
        dim, = normal.shape
        comps = []

        for d in range(dim):
            axis = principle_axis[d]
            for i in range(dim):
                comps.append(axis[i])

        comps = make_obj_array(comps)
        rotation_matrix = comps.reshape(dim, dim)

        # subtract the normal-component of velocity gradient to get the tangential part
        from mirgecom.fluid import velocity_gradient
        grad_v_minus = velocity_gradient(state_minus.cv, grad_cv_minus)

        # rotate the gradient tensor into the normal direction
        grad_v_minus_normal = rotation_matrix@grad_v_minus@rotation_matrix.T

        # set the shear terms in the plus state opposite the normal state to
        # cancel the shear flux
        grad_v_plus_shear = grad_v_minus_normal - grad_v_minus_normal*np.eye(state_minus.dim)
        grad_v_plus_normal = grad_v_minus_normal - 2*grad_v_plus_shear

        # get the gradient on the plus side in the original coordiate space
        grad_v_plus = rotation_matrix.T@grad_v_plus_normal@rotation_matrix

        # construct grad(mom)
        grad_mom_plus = (state_minus.mass_density*grad_v_plus
                         + np.outer(vel_plus, grad_cv_minus.mass))

        # modify grad(rhoE)
        # MJA, not needed? gradients are needed for the viscous fluxes and
        # energy does not appear there

        return make_conserved(grad_cv_minus.dim,
                              mass=grad_cv_minus.mass,
                              energy=grad_cv_minus.energy,
                              momentum=grad_mom_plus,
                              species_mass=grad_species_mass_plus)

    def viscous_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                          grad_cv_minus, grad_t_minus,
                          numerical_flux_func=viscous_facial_flux_central,
                          **kwargs):
        """Return the boundary flux for the divergence of the viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)

        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        state_wall = self.adiabatic_slip_wall_state(dcoll=dcoll, dd_bdry=dd_bdry,
                                                    gas_model=gas_model,
                                                    state_minus=state_minus,
                                                    **kwargs)

        grad_cv_wall = self.grad_cv_bc(state_minus=state_minus,
                                       state_plus=state_wall,
                                       grad_cv_minus=grad_cv_minus,
                                       normal=normal, **kwargs)

        grad_t_wall = self.grad_temperature_bc(
            grad_t_minus=grad_t_minus,
            normal=normal, **kwargs)

        state_pair = TracePair(dd_bdry, interior=state_minus,
                               exterior=state_wall)
        grad_cv_pair = TracePair(dd_bdry, interior=grad_cv_minus,
                                 exterior=grad_cv_wall)
        grad_t_pair = TracePair(dd_bdry, interior=grad_t_minus,
                                exterior=grad_t_wall)

        return (numerical_flux_func(dcoll, state_pair=state_pair,
                                    grad_cv_pair=grad_cv_pair,
                                    grad_t_pair=grad_t_pair,
                                    gas_model=gas_model))


def get_mesh(dim, size, bl_ratio, interface_ratio, angle=0.,
             transfinite=False, use_gmsh=False):
    """Generate a grid using `gmsh`.

    """

    height = 0.02
    fluid_length = 0.1
    wall_length = 0.02
    bottom_inflow = np.zeros(shape=(dim,))
    top_inflow = np.zeros(shape=(dim,))
    bottom_interface = np.zeros(shape=(dim,))
    top_interface = np.zeros(shape=(dim,))
    bottom_wall = np.zeros(shape=(dim,))
    top_wall = np.zeros(shape=(dim,))

    # rotate the mesh around the bottom-left corner
    theta = angle/180.*np.pi/2.
    bottom_inflow[0] = 0.0
    bottom_inflow[1] = -0.01
    top_inflow[0] = bottom_inflow[0] - height*np.sin(theta)
    top_inflow[1] = bottom_inflow[1] + height*np.cos(theta)

    bottom_interface[0] = bottom_inflow[0] + fluid_length*np.cos(theta)
    bottom_interface[1] = bottom_inflow[1] + fluid_length*np.sin(theta)
    top_interface[0] = top_inflow[0] + fluid_length*np.cos(theta)
    top_interface[1] = top_inflow[1] + fluid_length*np.sin(theta)

    bottom_wall[0] = bottom_interface[0] + wall_length*np.cos(theta)
    bottom_wall[1] = bottom_interface[1] + wall_length*np.sin(theta)
    top_wall[0] = top_interface[0] + wall_length*np.cos(theta)
    top_wall[1] = top_interface[1] + wall_length*np.sin(theta)

    if use_gmsh:
        from meshmode.mesh.io import (
            generate_gmsh,
            ScriptSource
        )

    # for 2D, the line segments/surfaces need to be specified clockwise to
        # get the correct facing (right-handed) surface normals
        my_string = (f"""
                Point(1) = {{ {bottom_inflow[0]},  {bottom_inflow[1]}, 0, {size}}};
                Point(2) = {{ {bottom_interface[0]}, {bottom_interface[1]},  0, {size}}};
                Point(3) = {{ {top_interface[0]}, {top_interface[1]},    0, {size}}};
                Point(4) = {{ {top_inflow[0]},  {top_inflow[1]},    0, {size}}};
                Point(5) = {{ {bottom_wall[0]},  {bottom_wall[1]},    0, {size}}};
                Point(6) = {{ {top_wall[0]},  {top_wall[1]},    0, {size}}};
                Line(1) = {{1, 2}};
                Line(2) = {{2, 3}};
                Line(3) = {{3, 4}};
                Line(4) = {{4, 1}};
                Line(5) = {{3, 6}};
                Line(6) = {{2, 5}};
                Line(7) = {{5, 6}};
                Line Loop(1) = {{-4, -3, -2, -1}};
                Line Loop(2) = {{2, 5, -7, -6}};
                Plane Surface(1) = {{1}};
                Plane Surface(2) = {{2}};
                Physical Surface('fluid') = {{1}};
                Physical Surface('solid') = {{2}};
                Physical Curve('fluid_inflow') = {{4}};
                Physical Curve('fluid_wall') = {{1,3}};
                Physical Curve('fluid_wall_top') = {{3}};
                Physical Curve('fluid_wall_bottom') = {{1}};
                Physical Curve('interface') = {{2}};
                Physical Curve('solid_wall') = {{5, 6, 7}};
                Physical Curve('solid_wall_top') = {{5}};
                Physical Curve('solid_wall_bottom') = {{6}};
                Physical Curve('solid_wall_end') = {{7}};
        """)

        if transfinite:
            my_string += (f"""

                    Transfinite Curve {{1, 3}} = {0.1} / {size};
                    Transfinite Curve {{5, 6}} = {0.02} / {size};
                    Transfinite Curve {{-2, 4, 7}} = {0.02} / {size} Using Bump 1/{bl_ratio};
                    Transfinite Surface {{1, 2}} Right;

                    Mesh.MeshSizeExtendFromBoundary = 0;
                    Mesh.MeshSizeFromPoints = 0;
                    Mesh.MeshSizeFromCurvature = 0;

                    Mesh.Algorithm = 5;
                    Mesh.OptimizeNetgen = 1;
                    Mesh.Smoothing = 0;
            """)
        else:
            my_string += (f"""
                    // Create distance field from curves, excludes cavity
                    Field[1] = Distance;
                    Field[1].CurvesList = {{1,3}};
                    Field[1].NumPointsPerCurve = 100000;

                    //Create threshold field that varrries element size near boundaries
                    Field[2] = Threshold;
                    Field[2].InField = 1;
                    Field[2].SizeMin = {size} / {bl_ratio};
                    Field[2].SizeMax = {size};
                    Field[2].DistMin = 0.0002;
                    Field[2].DistMax = 0.005;
                    Field[2].StopAtDistMax = 1;

                    //  background mesh size
                    Field[3] = Box;
                    Field[3].XMin = 0.;
                    Field[3].XMax = 1.0;
                    Field[3].YMin = -1.0;
                    Field[3].YMax = 1.0;
                    Field[3].VIn = {size};

                    // Create distance field from curves, excludes cavity
                    Field[4] = Distance;
                    Field[4].CurvesList = {{2}};
                    Field[4].NumPointsPerCurve = 100000;

                    //Create threshold field that varrries element size near boundaries
                    Field[5] = Threshold;
                    Field[5].InField = 4;
                    Field[5].SizeMin = {size} / {interface_ratio};
                    Field[5].SizeMax = {size};
                    Field[5].DistMin = 0.0002;
                    Field[5].DistMax = 0.005;
                    Field[5].StopAtDistMax = 1;

                    // take the minimum of all defined meshing fields
                    Field[100] = Min;
                    Field[100].FieldsList = {{2, 3, 5}};
                    Background Field = 100;

                    Mesh.MeshSizeExtendFromBoundary = 0;
                    Mesh.MeshSizeFromPoints = 0;
                    Mesh.MeshSizeFromCurvature = 0;

                    Mesh.Algorithm = 5;
                    Mesh.OptimizeNetgen = 1;
                    Mesh.Smoothing = 100;
            """)

        #print(my_string)
        generate_mesh = partial(generate_gmsh, ScriptSource(my_string, "geo"),
                                force_ambient_dim=2, dimensions=2, target_unit="M",
                                return_tag_to_elements_map=True)
    else:
        char_len_x = 0.002
        char_len_y = 0.001
        box_ll = (left_boundary_loc, bottom_boundary_loc)
        box_ur = (right_boundary_loc, top_boundary_loc)
        num_elements = (int((box_ur[0]-box_ll[0])/char_len_x),
                            int((box_ur[1]-box_ll[1])/char_len_y))

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh, a=box_ll, b=box_ur, n=num_elements,
          boundary_tag_to_face={
              "Inflow":["-x"],
              "Outflow":["+x"],
              "Wall":["+y","-y"]
              }
        )

    return generate_mesh

class InitSponge:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x0, thickness, amplitude):
        r"""Initialize the sponge parameters.

        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x0 = x0
        self._thickness = thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        x0 = zeros + self._x0

        return self._amplitude * actx.np.where(
            actx.np.greater(xpos, x0),
            (zeros + ((xpos - self._x0)/self._thickness) *
            ((xpos - self._x0)/self._thickness)),
            zeros + 0.0
        )


def getIsentropicPressure(mach, P0, gamma):
    pressure = (1. + (gamma - 1.)*0.5*mach**2)
    pressure = P0*pressure**(-gamma / (gamma - 1.))
    return pressure


def getIsentropicTemperature(mach, T0, gamma):
    temperature = (1. + (gamma - 1.)*0.5*mach**2)
    temperature = T0/temperature
    return temperature


def smooth_step(actx, x, epsilon=1e-12):
    # return actx.np.tanh(x)
    # return actx.np.where(
    #     actx.np.greater(x, 0),
    #     actx.np.tanh(x)**3,
    #     0*x)
    return (
        actx.np.greater(x, 0) * actx.np.less(x, 1) * (1 - actx.np.cos(np.pi*x))/2
        + actx.np.greater(x, 1))


class InitShock:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, nspecies=0,
            P0, T0, temp_wall, temp_sigma, vel_sigma, gamma_guess,
            mass_frac=None
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        P0: float
            stagnation pressure
        T0: float
            stagnation temperature
        gamma_guess: float
            guesstimate for gamma
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        """

        if mass_frac is None:
            if nspecies > 0:
                mass_frac = np.zeros(shape=(nspecies,))

        self._dim = dim
        self._nspecies = nspecies
        self._P0 = P0
        self._T0 = T0
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._gamma_guess = gamma_guess
        self._mass_frac = mass_frac

    def __call__(self, dcoll, x_vec, eos, *, time=0.0):
        """Create the solution state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        xpos = x_vec[0]
        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]
        ytop = 0*x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        ones = zeros + 1.0

        mach = zeros
        ytop = zeros
        ybottom = zeros
        theta = zeros
        gamma = self._gamma_guess

        pressure = getIsentropicPressure(
            mach=mach,
            P0=self._P0,
            gamma=gamma
        )
        temperature = getIsentropicTemperature(
            mach=mach,
            T0=self._T0,
            gamma=gamma
        )

        # save the unsmoothed temerature, so we can use it with the injector init
        unsmoothed_temperature = temperature

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma
        wall_temperature = self._temp_wall
        smoothing_top = smooth_step(actx, -sigma*(ypos-ytop))
        smoothing_bottom = smooth_step(
            actx, sigma*actx.np.abs(ypos-ybottom))
        smoothing_fore = ones
        smoothing_aft = ones
        z0 = 0.
        z1 = 0.035
        if self._dim == 3:
            smoothing_fore = smooth_step(actx, sigma*(zpos-z0))
            smoothing_aft = smooth_step(actx, -sigma*(zpos-z1))

        smooth_temperature = (wall_temperature +
            (temperature - wall_temperature)*smoothing_top*smoothing_bottom *
                                             smoothing_fore*smoothing_aft)

        y = ones*self._mass_frac
        mass = eos.get_density(pressure=pressure, temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        velocity = np.zeros(self._dim, dtype=object)
        mom = mass*velocity
        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom, energy=energy,
                            species_mass=mass*y)
        velocity[0] = mach*eos.sound_speed(cv, temperature)

        # modify the velocity in the near-wall region to have a smooth profile
        # this approximates the BL velocity profile
        sigma = self._vel_sigma
        smoothing_top = smooth_step(actx, -sigma*(ypos-ytop))
        smoothing_bottom = smooth_step(actx, sigma*(actx.np.abs(ypos-ybottom)))
        smoothing_fore = ones
        smoothing_aft = ones
        if self._dim == 3:
            smoothing_fore = smooth_step(actx, sigma*(zpos-z0))
            smoothing_aft = smooth_step(actx, -sigma*(zpos-z1))
        velocity[0] = (velocity[0]*smoothing_top*smoothing_bottom *
                       smoothing_fore*smoothing_aft)

        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )


def mask_from_elements(vol_discr, actx, elements):
    mesh = vol_discr.mesh
    zeros = vol_discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return DOFArray(actx, tuple(group_arrays))


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallVars:
    mass: DOFArray
    energy: DOFArray
    ox_mass: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (WallVars, tuple(getattr(self, f.name)
                                    for f in fields(WallVars)))


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    thermal_conductivity: DOFArray
    temperature: DOFArray


class WallModel:
    """Model for calculating wall quantities."""
    def __init__(
            self,
            heat_capacity,
            thermal_conductivity_func,
            *,
            effective_surface_area_func=None,
            mass_loss_func=None,
            oxygen_diffusivity=0.):
        self._heat_capacity = heat_capacity
        self._thermal_conductivity_func = thermal_conductivity_func
        self._effective_surface_area_func = effective_surface_area_func
        self._mass_loss_func = mass_loss_func
        self._oxygen_diffusivity = oxygen_diffusivity

    @property
    def heat_capacity(self):
        return self._heat_capacity

    def thermal_conductivity(self, mass, temperature):
        return self._thermal_conductivity_func(mass, temperature)

    def thermal_diffusivity(self, mass, temperature, thermal_conductivity=None):
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(mass, temperature)
        return thermal_conductivity/(mass * self.heat_capacity)

    def mass_loss_rate(self, mass, ox_mass, temperature):
        dm = mass*0.
        if self._effective_surface_area_func is not None:
            eff_surf_area = self._effective_surface_area_func(mass)
            if self._mass_loss_func is not None:
                dm = self._mass_loss_func(mass, ox_mass, temperature, eff_surf_area)
        return dm

    @property
    def oxygen_diffusivity(self):
        return self._oxygen_diffusivity

    def temperature(self, wv):
        return wv.energy/(wv.mass * self.heat_capacity)

    def dependent_vars(self, wv):
        temperature = self.temperature(wv)
        kappa = self.thermal_conductivity(wv.mass, temperature)
        return WallDependentVars(
            thermal_conductivity=kappa,
            temperature=temperature)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         restart_filename=None, target_filename=None,
         use_profiling=False, use_logmgr=True, user_input_file=None,
         use_overintegration=False, actx_class=None, casename=None,
         lazy=False):

    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    log_path = "log_data/"
    #log_path = ""
    logname = log_path + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    comm.Barrier()

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    # default i/o junk frequencies
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 1
    nlimit = 0
    # verbosity for what gets written to viz dumps, increase for more stuff
    viz_level = 1

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-8
    t_final = 1e-7
    current_t = 0
    t_start = 0.
    current_step = 0
    current_cfl = 1.0
    constant_cfl = False
    force_eval = True

    # default health status bounds
    health_pres_min = 1.0e-1
    health_pres_max = 2.0e6
    health_temp_min = 1.0
    health_temp_max = 5000.
    health_mass_frac_min = -10
    health_mass_frac_max = 10

    # discretization and model control
    order = 2
    quadrature_order = -1
    alpha_sc = 0.3
    s0_sc = -5.0
    kappa_sc = 0.5
    dim = 2
    inv_num_flux = "rusanov"
    periodic = False
    noslip = False
    adiabatic = False
    crossflow = False

    # material properties
    mu = 1.0e-5
    spec_diff = 1.e-4
    spec_diff = 0.
    mu_override = False  # optionally read in from input
    kappa_override = False  # optionally read in from input
    nspecies = 0
    pyro_temp_iter = 3  # for pyrometheus, number of newton iterations
    pyro_temp_tol = 1.e-4  # for pyrometheus, toleranace for temperature residual
    transport_type = 0

    # ambient fluid conditions
    #   100 Pa
    #   298 K
    #   rho = 1.77619667e-3 kg/m^3
    #   velocity = 0,0,0
    pres_bkrnd = 100
    temp_bkrnd = 300
    mach = 2.0

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630
    # for graphite
    wall_insert_rho = 1625
    wall_insert_cp = 770
    wall_insert_kappa = 247.5  # This seems high

    # Fiberform
    # wall_insert_rho = 183.6
    # wall_insert_cp = 710
    wall_insert_ox_diff = spec_diff

    # rhs control
    use_sponge = True
    use_wall_ox = True
    use_wall_mass = True

    # artificial viscosity control
    #    0 - none
    #    1 - physical viscosity based, div(velocity) indicator
    use_av = 0

    # species limiter
    #    0 - none
    #    2 - limit in pre-step
    use_species_limiter = 0

    sponge_sigma = 1.0

    # mesh parameters
    mesh_size=0.001
    bl_ratio = 3
    interface_ratio = 2
    transfinite = False
    mesh_angle = 0.

    # Filtering is implemented according to HW Sec. 5.3
    # The modal response function is e^-(alpha * eta ^ 2s), where
    # - alpha is a user parameter (defaulted like HW's)
    # - eta := (mode - N_c)/(N - N_c)
    # - N_c := cutoff mode ( = *filter_frac* x order)
    # - s := order of the filter (divided by 2)
    # Modes below N_c are unfiltered. Modes above Nc are weighted
    # by the modal response function described above.
    #
    # Two different filters can be used with the prediction driver.
    # 1) Solution filtering: filters the solution every *soln_nfilter* steps
    # 2) RHS filtering: filters the RHS every substep
    #
    # Turn on SOLUTION filtering by setting soln_nfilter > 0
    # Turn on RHS filtering by setting use_rhs_filter = 1.
    #
    # --- Filtering settings ---
    # ------ Solution filtering
    soln_nfilter = -1  # filter every *nfilter* steps (-1 = no filtering)
    soln_filter_cutoff = -1  # (-1 = filter_frac*order)
    soln_filter_frac = .5
    soln_filter_order = 8

    # Alpha value suggested by:
    # JSH/TW Nodal DG Methods, Section 5.3
    # DOI: 10.1007/978-0-387-72067-8
    soln_filter_alpha = -1.0*np.log(np.finfo(float).eps)
    # ------ RHS filtering
    use_rhs_filter = False
    rhs_filter_cutoff = -1
    rhs_filter_frac = .5
    rhs_filter_order = 8
    rhs_filter_alpha = soln_filter_alpha

    # init params
    disc_location = np.zeros(shape=(dim,))
    shock_loc_x = 0.05

    fuel_location = np.zeros(shape=(dim,))
    fuel_loc_x = 0.07

    # parameters to adjust the shape of the initialization
    vel_sigma = 1000
    temp_sigma = 1250
    temp_wall = 300

    # wall stuff
    wall_penalty_amount = 25
    wall_time_scale = 50
    wall_material = 0

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        input_data = comm.bcast(input_data, root=0)
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            viz_level = int(input_data["viz_level"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            nlimit = int(input_data["nlimit"])
        except KeyError:
            pass
        try:
            periodic = bool(input_data["periodic"])
        except KeyError:
            pass
        try:
            crossflow = bool(input_data["crossflow"])
        except KeyError:
            pass
        try:
            use_species_limiter = int(input_data["use_species_limiter"])
        except KeyError:
            pass
        try:
            constant_cfl = int(input_data["constant_cfl"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            current_cfl = float(input_data["current_cfl"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            alpha_sc = float(input_data["alpha_sc"])
        except KeyError:
            pass
        try:
            kappa_sc = float(input_data["kappa_sc"])
        except KeyError:
            pass
        try:
            s0_sc = float(input_data["s0_sc"])
        except KeyError:
            pass
        try:
            mu_input = float(input_data["mu"])
            mu_override = True
        except KeyError:
            pass
        try:
            kappa_input = float(input_data["kappa"])
            kappa_override = True
        except KeyError:
            pass
        try:
            spec_diff = float(input_data["spec_diff"])
            wall_insert_ox_diff = spec_diff
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            order = int(input_data["quadrature_order"])
        except KeyError:
            pass
        try:
            dim = int(input_data["dimen"])
        except KeyError:
            pass
        try:
            nspecies = int(input_data["nspecies"])
        except KeyError:
            pass
        try:
            transport_type = int(input_data["transport"])
        except KeyError:
            pass
        try:
            pyro_temp_iter = int(input_data["pyro_temp_iter"])
        except KeyError:
            pass
        try:
            pyro_temp_tol = float(input_data["pyro_temp_tol"])
        except KeyError:
            pass
        try:
            vel_sigma = float(input_data["vel_sigma"])
        except KeyError:
            pass
        try:
            temp_sigma = float(input_data["temp_sigma"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            inv_num_flux = input_data["inviscid_numerical_flux"]
        except KeyError:
            pass
        try:
            health_pres_min = float(input_data["health_pres_min"])
        except KeyError:
            pass
        try:
            health_pres_max = float(input_data["health_pres_max"])
        except KeyError:
            pass
        try:
            health_temp_min = float(input_data["health_temp_min"])
        except KeyError:
            pass
        try:
            health_temp_max = float(input_data["health_temp_max"])
        except KeyError:
            pass
        try:
            health_mass_frac_min = float(input_data["health_mass_frac_min"])
        except KeyError:
            pass
        try:
            health_mass_frac_max = float(input_data["health_mass_frac_max"])
        except KeyError:
            pass
        try:
            use_sponge = bool(input_data["use_sponge"])
        except KeyError:
            pass
        try:
            sponge_sigma = float(input_data["sponge_sigma"])
        except KeyError:
            pass
        try:
            use_av = int(input_data["use_av"])
        except KeyError:
            pass
        try:
            use_wall_ox = bool(input_data["use_wall_ox"])
        except KeyError:
            pass
        try:
            use_wall_mass = bool(input_data["use_wall_mass"])
        except KeyError:
            pass
        try:
            mesh_filename = input_data["mesh_filename"]
        except KeyError:
            pass
        try:
            wall_penalty_amount = float(input_data["wall_penalty_amount"])
        except KeyError:
            pass
        try:
            wall_time_scale = float(input_data["wall_time_scale"])
        except KeyError:
            pass
        try:
            wall_material = int(input_data["wall_material"])
        except KeyError:
            pass
        try:
            wall_insert_rho = float(input_data["wall_insert_rho"])
        except KeyError:
            pass
        try:
            wall_insert_cp = float(input_data["wall_insert_cp"])
        except KeyError:
            pass
        try:
            wall_insert_kappa = float(input_data["wall_insert_kappa"])
        except KeyError:
            pass
        try:
            mesh_size = float(input_data["mesh_size"])
        except KeyError:
            pass
        try:
            mesh_angle = float(input_data["mesh_angle"])
        except KeyError:
            pass
        try:
            bl_ratio = float(input_data["bl_ratio"])
        except KeyError:
            pass
        try:
            interface_ratio = float(input_data["interface_ratio"])
        except KeyError:
            pass
        try:
            transfinite = bool(input_data["transfinite"])
        except KeyError:
            pass
        try:
            noslip = bool(input_data["noslip"])
        except KeyError:
            pass
        try:
            adiabatic = bool(input_data["adiabatic"])
        except KeyError:
            pass
        try:
            shock_loc_x = float(input_data["shock_loc"])
        except KeyError:
            pass
        try:
            fuel_loc_x = float(input_data["fuel_loc"])
        except KeyError:
            pass
        try:
            pres_bkrnd = float(input_data["pres_bkrnd"])
        except KeyError:
            pass
        try:
            temp_bkrnd = float(input_data["temp_bkrnd"])
        except KeyError:
            pass
        try:
            mach = float(input_data["mach"])
        except KeyError:
            pass
        try:
            soln_nfilter = int(input_data["soln_nfilter"])
        except KeyError:
            pass
        try:
            soln_filter_frac = float(input_data["soln_filter_frac"])
        except KeyError:
            pass
        try:
            soln_filter_cutoff = int(input_data["soln_filter_cutoff"])
        except KeyError:
            pass
        try:
            soln_filter_alpha = float(input_data["soln_filter_alpha"])
        except KeyError:
            pass
        try:
            soln_filter_order = int(input_data["soln_filter_order"])
        except KeyError:
            pass
        try:
            use_rhs_filter = bool(input_data["use_rhs_filter"])
        except KeyError:
            pass
        try:
            rhs_filter_frac = float(input_data["rhs_filter_frac"])
        except KeyError:
            pass
        try:
            rhs_filter_cutoff = int(input_data["rhs_filter_cutoff"])
        except KeyError:
            pass
        try:
            rhs_filter_alpha = float(input_data["rhs_filter_alpha"])
        except KeyError:
            pass
        try:
            rhs_filter_order = int(input_data["rhs_filter_order"])
        except KeyError:
            pass

    disc_location[0] = shock_loc_x
    fuel_location[0] = fuel_loc_x

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144", "compiled_lsrk54"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if integrator == "compiled_lsrk54":
        print("Setting force_eval = False for pre-compiled time integration")
        force_eval = False

    s0_sc = np.log10(1.0e-4 / np.power(order, 4))
    if rank == 0:
        print(f"Shock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")

    if quadrature_order < 0:
        quadrature_order = 2*order + 1

    # use_av=3 specific parameters
    # flow stagnation temperature
    static_temp = 2076.43
    # steepness of the smoothed function
    theta_sc = 100
    # cutoff, smoothness below this value is ignored
    beta_sc = 0.01
    gamma_sc = 1.5

    if soln_filter_cutoff < 0:
        soln_filter_cutoff = int(soln_filter_frac * order)
    if rhs_filter_cutoff < 0:
        rhs_filter_cutoff = int(rhs_filter_frac * order)

    from mirgecom.filter import (
        exponential_mode_response_function as xmrfunc,
        filter_modally
    )
    soln_frfunc = partial(xmrfunc, alpha=soln_filter_alpha,
                          filter_order=soln_filter_order)
    rhs_frfunc = partial(xmrfunc, alpha=rhs_filter_alpha,
                         filter_order=rhs_filter_order)

    def filter_cv(cv):
        return filter_modally(dcoll, soln_filter_cutoff, soln_frfunc, cv,
                              dd=dd_vol_fluid)

    def filter_rhs(rhs):
        return filter_modally(dcoll, rhs_filter_cutoff, rhs_frfunc, rhs,
                              dd=dd_vol_fluid)

    filter_cv_compiled = actx.compile(filter_cv)

    if rank == 0:
        if soln_nfilter >= 0:
            if soln_filter_cutoff >= order:
                raise ValueError("Invalid setting for solution filter (cutoff >= order).")
            print("Solution filtering settings:")
            print(f" - filter every {soln_nfilter} steps")
            print(f" - filter alpha  = {soln_filter_alpha}")
            print(f" - filter cutoff = {soln_filter_cutoff}")
            print(f" - filter order  = {soln_filter_order}")
        else:
            print("Solution filtering OFF.")
        if use_rhs_filter:
            if rhs_filter_cutoff >= order:
                raise ValueError("Invalid setting for RHS filter (cutoff >= order).")
            print("RHS filtering settings:")
            print(f" - filter alpha  = {rhs_filter_alpha}")
            print(f" - filter cutoff = {rhs_filter_cutoff}")
            print(f" - filter order  = {rhs_filter_order}")
        else:
            print("RHS filtering OFF.")

    if rank == 0:
        if use_av == 0:
            print("Artificial viscosity disabled")
        else:
            print("Artificial viscosity using modified physical viscosity")
            print("Using velocity divergence indicator")
            print(f"Shock capturing parameters: alpha {alpha_sc}, "
                  f"gamma_sc {gamma_sc}"
                  f"theta_sc {theta_sc}, beta_sc {beta_sc}, Pr 0.75, "
                  f"stagnation temperature {static_temp}")

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if constant_cfl == 1:
            print(f"\tConstant cfl mode, current_cfl = {current_cfl}")
        else:
            print(f"\tConstant dt mode, current_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        if use_overintegration:
            print(f"\tOverintegration ON: quadrature_order = {quadrature_order}")
        print(f"\tdimen = {dim}")
        print(f"\tTime integration {integrator}")
        print("#### Simluation control data: ####\n")

    if rank == 0:
        print("\n#### Visualization setup: ####")
        if viz_level >= 0:
            print("\tBasic visualization output enabled.")
            print("\t(cv, dv, cfl)")
        if viz_level >= 1:
            print("\tExtra visualization output enabled for derived quantities.")
            print("\t(velocity, mass_fractions, etc.)")
        if viz_level >= 2:
            print("\tNon-dimensional parameter visualization output enabled.")
            print("\t(Re, Pr, etc.)")
        if viz_level >= 3:
            print("\tDebug visualization output enabled.")
            print("\t(rhs, grad_cv, etc.)")
        print("#### Visualization setup: ####")

    if rank == 0:
        print("\n#### Simluation setup data: ####")
        print(f"\tvel_sigma = {vel_sigma}")
        print(f"\ttemp_sigma = {temp_sigma}")
        print("#### Simluation setup data: ####")

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step
    if integrator == "compiled_lsrk54":
        timestepper = _compiled_stepper_wrapper

    if inv_num_flux == "rusanov":
        inviscid_numerical_flux_func = inviscid_facial_flux_rusanov
        if rank == 0:
            print("\nRusanov inviscid flux")
    if inv_num_flux == "hll":
        inviscid_numerical_flux_func = inviscid_facial_flux_hll
        if rank == 0:
            print("\nHLL inviscid flux")

    # }}}
    # working gas: O2/N2 #
    #   O2 mass fraction 0.273
    #   gamma = 1.4
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma = 1.4
    mw_o = 15.999
    mw_o2 = mw_o*2
    mw_co = 28.010
    mw_n2 = 14.0067*2
    mf_o2 = 0.273
    # visocsity @ 400C, Pa-s
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu_mix = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    univ_gas_const = 8314.59
    r = univ_gas_const/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75
    #Pr = 1000000000

    if mu_override:
        mu = mu_input
    else:
        mu = mu_mix

    kappa = cp*mu/Pr
    init_temperature = 300.0

    if kappa_override:
        kappa = kappa_input

    # don't allow limiting on flows without species
    if nspecies == 0:
        use_species_limiter = 0

    species_limit_sigma = 0
    if nlimit > 0:
        species_limit_sigma = 1./nlimit/current_dt

    # crossflow only in periodic cased
    if not periodic:
        crossflow = False
        print("\tSetting crossflow False for non-periodic case")

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print("#### Fluid domain: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")
        print(f"\tnspecies = {nspecies}")
        if nspecies == 0:
            print("\tno passive scalars, uniform ideal gas eos")
        elif nspecies == 3:
            print("\tfull multi-species initialization with pyrometheus eos")
        else:
            print("\tfull multi-species initialization with pyrometheus eos")

        if use_species_limiter == 1:
            print("\nSpecies mass fractions limited to [0:1]")
            print(f" outside the rhs every {nlimit} steps")
        elif use_species_limiter != 0:
            error_message = "Unknown species_limiting specification "
            raise RuntimeError(error_message)

    transport_alpha = 0.6
    transport_beta = 4.093e-7
    transport_sigma = 2.0
    transport_n = 0.666

    if rank == 0:
        if transport_type == 0:
            print("\t Simple transport model:")
            print("\t\t constant viscosity, species diffusivity")
            print(f"\tmu = {mu}")
            print(f"\tkappa = {kappa}")
            print(f"\tspecies diffusivity = {spec_diff}")
        elif transport_type == 1:
            print("\t Power law transport model:")
            print("\t\t temperature dependent viscosity, species diffusivity")
            print(f"\ttransport_alpha = {transport_alpha}")
            print(f"\ttransport_beta = {transport_beta}")
            print(f"\ttransport_sigma = {transport_sigma}")
            print(f"\ttransport_n = {transport_n}")
            print(f"\tspecies diffusivity = {spec_diff}")
        elif transport_type == 2:
            print("\t Pyrometheus transport model:")
            print("\t\t temperature/mass fraction dependence")
        else:
            error_message = "Unknown transport_type {}".format(transport_type)
            raise RuntimeError(error_message)

        print("#### Wall domain: ####")

        if wall_material == 0:
            print("\tNon-reactive wall model")
        elif wall_material == 1:
            print("\tReactive wall model for non-porous media")
        elif wall_material == 2:
            print("\tReactive wall model for porous media")
        else:
            error_message = "Unknown wall_material {}".format(wall_material)
            raise RuntimeError(error_message)

        if use_wall_ox:
            print("\tWall oxidizer transport enabled")
        else:
            print("\tWall oxidizer transport disabled")

        if use_wall_mass:
            print("\t Wall mass loss enabled")
        else:
            print("\t Wall mass loss disabled")

        print(f"\tWall density = {wall_insert_rho}")
        print(f"\tWall cp = {wall_insert_cp}")
        print(f"\tWall O2 diff = {wall_insert_ox_diff}")
        print(f"\tWall time scale = {wall_time_scale}")
        print(f"\tWall penalty = {wall_penalty_amount}")
        print("#### Simluation material properties: ####")

    spec_diffusivity = spec_diff * np.ones(nspecies)
    if transport_type == 0:
        physical_transport_model = SimpleTransport(
            viscosity=mu, thermal_conductivity=kappa,
            species_diffusivity=spec_diffusivity)
    if transport_type == 1:
        physical_transport_model = PowerLawTransport(
            alpha=transport_alpha, beta=transport_beta,
            sigma=transport_sigma, n=transport_n,
            species_diffusivity=spec_diffusivity)

    if use_av == 1:
        transport_model = ArtificialViscosityTransportDiv(
            physical_transport=physical_transport_model,
            av_mu=alpha_sc, av_prandtl=0.75)
    else:
        transport_model = physical_transport_model

    rho_bkrnd = pres_bkrnd/r/temp_bkrnd
    c_bkrnd = math.sqrt(gamma*pres_bkrnd/rho_bkrnd)

    pressure_ratio = (2.*gamma*mach*mach-(gamma-1.))/(gamma+1.)
    density_ratio = (gamma+1.)*mach*mach/((gamma-1.)*mach*mach+2.)
    #mach2 = math.sqrt(((gamma-1.)*mach*mach+2.)/(2.*gamma*mach*mach-(gamma-1.)))

    rho1 = rho_bkrnd
    pressure1 = pres_bkrnd
    temperature1 = pressure1/rho1/r
    rho2 = rho1*density_ratio
    pressure2 = pressure1*pressure_ratio
    temperature2 = pressure2/rho2/r
    velocity2 = -mach*c_bkrnd*(1/density_ratio-1)
    temp_wall = temperature1

    vel_left = np.zeros(shape=(dim,))
    vel_right = np.zeros(shape=(dim,))
    vel_cross = np.zeros(shape=(dim,))
    vel_cross[1] = 0

    if rank == 0:
        print("#### Simluation initialization data: ####")
        print(f"\tShock Mach number {mach}")
        print(f"\tgamma {gamma}")
        print(f"\tambient temperature {temperature1}")
        print(f"\tambient pressure {pressure1}")
        print(f"\tambient rho {rho1}")
        print(f"\tambient velocity {vel_right[0]}")
        print(f"\tpost-shock temperature {temperature2}")
        print(f"\tpost-shock pressure {pressure2}")
        print(f"\tpost-shock rho {rho2}")
        print(f"\tpost-shock velocity {velocity2}")

    plane_normal = np.zeros(shape=(dim,))
    theta = mesh_angle/180.*np.pi/2.
    plane_normal[0] = np.cos(theta)
    plane_normal[1] = np.sin(theta)
    plane_normal = plane_normal/np.linalg.norm(plane_normal)

    vel_left = velocity2*plane_normal

    chem_source_tol = 1.e-10
    # make the eos
    if nspecies < 3:
        eos = IdealSingleGas(gamma=gamma, gas_const=r)
        species_names = ["air", "fuel"]
    else:
        from mirgecom.thermochemistry import get_pyrometheus_wrapper_class
        from uiuc import Thermochemistry
        pyro_mech = get_pyrometheus_wrapper_class(
            pyro_class=Thermochemistry, temperature_niter=pyro_temp_iter,
            zero_level=chem_source_tol)(actx.np)
        eos = PyrometheusMixture(pyro_mech, temperature_guess=init_temperature)
        species_names = pyro_mech.species_names

    gas_model = GasModel(eos=eos, transport=transport_model)

    # initialize eos and species mass fractions
    y = np.zeros(nspecies)
    y_fuel = np.zeros(nspecies)
    if nspecies == 7:
        # find name species indicies
        for i in range(nspecies):
            if species_names[i] == "O2":
                i_ox = i
            if species_names[i] == "N2":
                i_di = i
            if species_names[i] == "C2H4":
                i_fuel = i

        # Set the species mass fractions to the free-stream flow
        y[i_ox] = mf_o2
        y[i_di] = 1. - mf_o2
        y_fuel[i_fuel] = 1.

    bulk_init = PlanarDiscontinuityMulti(dim=dim,
                                         nspecies=nspecies,
                                         disc_location=disc_location,
                                         disc_location_species=fuel_location,
                                         normal_dir=plane_normal,
                                         sigma=0.001,
                                         pressure_left=pressure2,
                                         pressure_right=pressure1,
                                         temperature_left=temperature2,
                                         temperature_right=temperature1,
                                         velocity_left=vel_left,
                                         velocity_right=vel_right,
                                         velocity_cross=vel_cross,
                                         species_mass_left=y,
                                         species_mass_right=y_fuel,
                                         temp_wall=temp_bkrnd,
                                         vel_sigma=vel_sigma,
                                         temp_sigma=temp_sigma)

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    )

    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        t_start = current_t
        volume_to_local_mesh_data = restart_data["volume_to_local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert restart_data["nparts"] == nparts
        assert restart_data["nspecies"] == nspecies
    else:  # generate the grid from scratch
        if rank == 0:
            print("Generating mesh")

        def get_mesh_data():
            #from meshmode.mesh.io import read_gmsh
            #mesh, tag_to_elements = read_gmsh(
                #mesh_filename, force_ambient_dim=dim,
                #return_tag_to_elements_map=True)
            mesh, tag_to_elements = get_mesh(dim=dim, angle=mesh_angle, use_gmsh=True, size=mesh_size,
                                             bl_ratio=bl_ratio,
                                             interface_ratio=interface_ratio,
                                             transfinite=transfinite)()
            volume_to_tags = {
                "fluid": ["fluid"],
                "wall": ["solid"]}

            # apply periodicity
            if periodic:

                from meshmode.mesh.processing import (
                    glue_mesh_boundaries, BoundaryPairMapping)

                from meshmode import AffineMap
                bdry_pair_mappings_and_tols = []
                offset = [0., 0.02]
                bdry_pair_mappings_and_tols.append((
                    BoundaryPairMapping(
                        "fluid_wall_bottom",
                        "fluid_wall_top",
                        AffineMap(offset=offset)),
                    1e-12))

                bdry_pair_mappings_and_tols.append((
                    BoundaryPairMapping(
                        "solid_wall_bottom",
                        "solid_wall_top",
                        AffineMap(offset=offset)),
                    1e-12))

                mesh = glue_mesh_boundaries(mesh, bdry_pair_mappings_and_tols)

            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

    local_nelements = (
        volume_to_local_mesh_data["fluid"][0].nelements
        + volume_to_local_mesh_data["wall"][0].nelements)

    # target data, used for sponge and prescribed boundary condtitions
    if target_filename:  # read the grid from restart data
        target_filename = f"{target_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        target_data = read_restart_data(actx, target_filename)
        #volume_to_local_mesh_data = target_data["volume_to_local_mesh_data"]
        global_nelements = target_data["global_nelements"]
        target_order = int(target_data["order"])

        assert target_data["nparts"] == nparts
        assert target_data["nspecies"] == nspecies
        assert target_data["global_nelements"] == global_nelements
    else:
        logger.warning("No target file specied, using restart as target")

    if rank == 0:
        logger.info("Making discretization")

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    dcoll = \
        create_discretization_collection(
            actx,
            volume_meshes={
                vol: mesh
                for vol, (mesh, _) in volume_to_local_mesh_data.items()},
            order=order, quadrature_order=quadrature_order
        )

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_wall = DOFDesc(VolumeDomainTag("wall"), DISCR_TAG_BASE)

    wall_vol_discr = dcoll.discr_from_dd(dd_vol_wall)
    wall_tag_to_elements = volume_to_local_mesh_data["wall"][1]
    wall_mask = mask_from_elements(
        wall_vol_discr, actx, wall_tag_to_elements["solid"])

    from grudge.dt_utils import characteristic_lengthscales
    char_length = characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid)
    char_length_wall = characteristic_lengthscales(actx, dcoll, dd=dd_vol_wall)

    # some utility functions
    def vol_min_loc(dd_vol, x):
        return actx.to_numpy(op.nodal_min_loc(dcoll, dd_vol, x,
                                              initial=np.inf))[()]

    def vol_max_loc(dd_vol, x):
        return actx.to_numpy(op.nodal_max_loc(dcoll, dd_vol, x,
                                              initial=-np.inf))[()]

    def vol_min(dd_vol, x):
        return actx.to_numpy(op.nodal_min(dcoll, dd_vol, x,
                                          initial=np.inf))[()]

    def vol_max(dd_vol, x):
        return actx.to_numpy(op.nodal_max(dcoll, dd_vol, x,
                                          initial=-np.inf))[()]

    def global_range_check(dd_vol, array, min_val, max_val):
        return global_reduce(
            check_range_local(
                dcoll, dd_vol, array, min_val, max_val), op="lor")

    h_min_fluid = vol_min(dd_vol_fluid, char_length)
    h_max_fluid = vol_max(dd_vol_fluid, char_length)
    # h_min_wall = vol_min(dd_vol_wall, char_length_wall)
    # h_max_wall = vol_max(dd_vol_wall, char_length_wall)

    if rank == 0:
        print(f"{h_min_fluid=},{h_max_fluid=}")

    if rank == 0:
        logger.info("Before restart/init")

    #########################
    # Convenience Functions #
    #########################

    def _create_fluid_state(cv, temperature_seed, smoothness=None):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature_seed,
                                smoothness=smoothness)

    create_fluid_state = actx.compile(_create_fluid_state)

    def update_dv(cv, temperature, smoothness):
        from mirgecom.eos import MixtureDependentVars, GasDependentVars
        if nspecies < 3:
            return GasDependentVars(
                temperature=temperature,
                pressure=eos.pressure(cv, temperature),
                speed_of_sound=eos.sound_speed(cv, temperature),
                smoothness=smoothness)
        else:
            return MixtureDependentVars(
                temperature=temperature,
                pressure=eos.pressure(cv, temperature),
                speed_of_sound=eos.sound_speed(cv, temperature),
                species_enthalpies=eos.species_enthalpies(cv, temperature),
                smoothness=smoothness)

    def update_tv(cv, dv):
        return gas_model.transport.transport_vars(cv, dv, eos)

    def update_fluid_state(cv, dv, tv):
        from mirgecom.gas_model import ViscousFluidState
        return ViscousFluidState(cv, dv, tv)

    update_dv_compiled = actx.compile(update_dv)
    update_tv_compiled = actx.compile(update_tv)
    update_fluid_state_compiled = actx.compile(update_fluid_state)

    def _create_wall_dependent_vars(wv):
        return wall_model.dependent_vars(wv)

    create_wall_dependent_vars_compiled = actx.compile(
        _create_wall_dependent_vars)

    def _get_wv(wv):
        return wv

    get_wv = actx.compile(_get_wv)

    def get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = gas_model.eos.internal_energy(cv)/cv.mass
        return actx.np.abs(
            pyro_mech.get_temperature_update_energy(e, temperature, y))

    get_temperature_update_compiled = actx.compile(get_temperature_update)

    def compute_smoothness(cv, dv, grad_cv):

        from mirgecom.fluid import velocity_gradient
        div_v = np.trace(velocity_gradient(cv, grad_cv))

        gamma = gas_model.eos.gamma(cv=cv, temperature=dv.temperature)
        r = gas_model.eos.gas_const(cv)
        c_star = actx.np.sqrt(gamma*r*(2/(gamma+1)*static_temp))
        indicator = -gamma_sc*char_length*div_v/c_star

        smoothness = actx.np.log(
            1 + actx.np.exp(theta_sc*(indicator - beta_sc)))/theta_sc
        return smoothness*gamma_sc*char_length

    compute_smoothness_compiled = actx.compile(compute_smoothness) # noqa

    if restart_filename:
        if rank == 0:
            logger.info("Restarting soln.")
        temperature_seed = restart_data["temperature_seed"]
        restart_cv = restart_data["cv"]
        restart_wv = restart_data["wv"]
        if restart_order != order:
            restart_dcoll = create_discretization_collection(
                actx,
                volume_meshes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=restart_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_fluid),
                restart_dcoll.discr_from_dd(dd_vol_fluid)
            )
            wall_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_wall),
                restart_dcoll.discr_from_dd(dd_vol_wall)
            )
            restart_cv = fluid_connection(restart_data["cv"])
            temperature_seed = fluid_connection(restart_data["temperature_seed"])
            restart_wv = wall_connection(restart_data["wv"])

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        if rank == 0:
            logger.info("Initializing soln.")
        restart_cv = bulk_init(
            x_vec=actx.thaw(dcoll.nodes(dd_vol_fluid)), eos=eos,
            time=0)
        temperature_seed = 0*restart_cv.mass + init_temperature
        wall_mass = wall_insert_rho * wall_mask
        wall_cp = wall_insert_cp * wall_mask
        restart_wv = WallVars(
            mass=wall_mass,
            energy=wall_mass * wall_cp * temp_wall,
            ox_mass=0*wall_mass)

    if target_filename:
        if rank == 0:
            logger.info("Reading target soln.")
        if target_order != order:
            target_dcoll = create_discretization_collection(
                actx,
                volume_meshes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=target_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_fluid),
                target_dcoll.discr_from_dd(dd_vol_fluid)
            )
            target_cv = fluid_connection(target_data["cv"])
        else:
            target_cv = target_data["cv"]
    else:
        # Set the current state from time 0
        target_cv = restart_cv

    no_smoothness = force_evaluation(actx, 0.*restart_cv.mass)
    smoothness = no_smoothness
    target_smoothness = smoothness

    restart_cv = force_evaluation(actx, restart_cv)
    target_cv = force_evaluation(actx, target_cv)
    temperature_seed = force_evaluation(actx, temperature_seed)

    current_fluid_state = create_fluid_state(restart_cv, temperature_seed,
                                             smoothness=smoothness)
    target_fluid_state = create_fluid_state(target_cv, temperature_seed,
                                            smoothness=target_smoothness)
    current_wv = force_evaluation(actx, restart_wv)
    #current_wv = get_wv(restart_wv)

    if noslip:
        if adiabatic:
            fluid_wall = AdiabaticNoslipWallBoundary()
        else:
            fluid_wall = IsothermalWallBoundary(temp_wall)

    else:
        # new implementation, following Mengaldo with modifications for slip vs no slip
        # tries to set the flux directly, instead of cancelling through the numerical viscous flux
        #fluid_wall = AdiabaticSlipWallBoundary2()

        # implementation from mirgecom
        # should be same as AdiabaticSlipBoundary2 
        fluid_wall = AdiabaticSlipBoundary()

        # new implementation, following Mengaldo with modifications for slip vs no slip
        # local version
        #fluid_wall = AdiabaticSlipWallBoundary()


        # Tulio's symmetry boundary
        #fluid_wall = SymmetryBoundary(dim=dim)

    wall_farfield = DirichletDiffusionBoundary(temp_wall)

    # use dummy boundaries to setup the smoothness state for the target
    target_boundaries = {
        dd_vol_fluid.trace("fluid_inflow").domain_tag: DummyBoundary(),
        #dd_vol_fluid.trace("fluid_wall").domain_tag: IsothermalWallBoundary()
        dd_vol_fluid.trace("fluid_wall").domain_tag: fluid_wall
    }

    def _grad_cv_operator_target(fluid_state, time):
        return grad_cv_operator(dcoll=dcoll, gas_model=gas_model,
                                dd=dd_vol_fluid,
                                boundaries=target_boundaries,
                                state=fluid_state,
                                time=time,
                                quadrature_tag=quadrature_tag)

    grad_cv_operator_target_compiled = actx.compile(_grad_cv_operator_target) # noqa

    if use_av:
        target_grad_cv = grad_cv_operator_target_compiled(
            target_fluid_state, time=0.)
        target_smoothness = compute_smoothness_compiled(
            cv=target_cv, dv=target_fluid_state.dv, grad_cv=target_grad_cv)

        target_fluid_state = create_fluid_state(cv=target_cv,
                                          temperature_seed=temperature_seed,
                                          smoothness=target_smoothness)

    stepper_state = make_obj_array([current_fluid_state.cv,
                                    temperature_seed, current_wv])

    ##################################
    # Set up the boundary conditions #
    ##################################

    from mirgecom.gas_model import project_fluid_state

    def get_target_state_on_boundary(btag):
        return project_fluid_state(
            dcoll, dd_vol_fluid,
            dd_vol_fluid.trace(btag).with_discr_tag(quadrature_tag),
            target_fluid_state, gas_model
        )

    flow_ref_state = \
        get_target_state_on_boundary("fluid_inflow")

    flow_ref_state = force_evaluation(actx, flow_ref_state)

    def _target_flow_state_func(**kwargs):
        return flow_ref_state

    flow_boundary = PrescribedFluidBoundary(
        boundary_state_func=_target_flow_state_func)

    if periodic:
        fluid_boundaries = {
            dd_vol_fluid.trace("fluid_inflow").domain_tag: flow_boundary,
        }

        wall_boundaries = {
            dd_vol_wall.trace("solid_wall_end").domain_tag: wall_farfield
        }
    else:
        fluid_boundaries = {
            dd_vol_fluid.trace("fluid_inflow").domain_tag: flow_boundary,
            dd_vol_fluid.trace("fluid_wall").domain_tag: fluid_wall
        }

        wall_boundaries = {
            dd_vol_wall.trace("solid_wall").domain_tag: wall_farfield
        }

    # compiled wrapper for grad_cv_operator
    def _grad_cv_operator(fluid_state, time):
        return grad_cv_operator(dcoll=dcoll, gas_model=gas_model,
                                boundaries=fluid_boundaries,
                                dd=dd_vol_fluid,
                                state=fluid_state,
                                time=time,
                                quadrature_tag=quadrature_tag)

    grad_cv_operator_compiled = actx.compile(_grad_cv_operator) # noqa

    def get_production_rates(cv, temperature):
        return eos.get_production_rates(cv, temperature)

    compute_production_rates = actx.compile(get_production_rates)

    def _grad_t_operator(t, fluid_state, wall_kappa, wall_temperature):
        fluid_grad_t, wall_grad_t = coupled_grad_t_operator(
            dcoll,
            gas_model,
            dd_vol_fluid, dd_vol_wall,
            fluid_boundaries, wall_boundaries,
            fluid_state, wall_kappa, wall_temperature,
            time=t,
            quadrature_tag=quadrature_tag)
        return make_obj_array([fluid_grad_t, wall_grad_t])

    grad_t_operator = actx.compile(_grad_t_operator)

    ##################
    # Sponge Sources #
    ##################

    # initialize the sponge field
    sponge_thickness = 0.09
    sponge_amp = sponge_sigma/current_dt/1000
    sponge_x0 = 0.9

    sponge_init = InitSponge(x0=sponge_x0, thickness=sponge_thickness,
                             amplitude=sponge_amp)
    x_vec = actx.thaw(dcoll.nodes(dd_vol_fluid))

    def _sponge_sigma(x_vec):
        return sponge_init(x_vec=x_vec)

    get_sponge_sigma = actx.compile(_sponge_sigma)
    sponge_sigma = get_sponge_sigma(x_vec)

    def _sponge_source(cv):
        """Create sponge source."""
        return sponge_sigma*(current_fluid_state.cv - cv)

    def experimental_kappa(temperature):
        return (
            1.766e-10 * temperature**3
            - 4.828e-7 * temperature**2
            + 6.252e-4 * temperature
            + 6.707e-3)

    def puma_kappa(mass_loss_frac):
        return (
            0.0988 * mass_loss_frac**2
            - 0.2751 * mass_loss_frac
            + 0.201)

    def puma_effective_surface_area(mass_loss_frac):
        # Original fit function: -1.1012e5*x**2 - 0.0646e5*x + 1.1794e5
        # Rescale by x==0 value and rearrange
        return 1.1794e5 * (
            1
            - 0.0547736137 * mass_loss_frac
            - 0.9336950992 * mass_loss_frac**2)

    def _get_wall_kappa_fiber(mass, temperature):
        mass_loss_frac = (wall_insert_rho - mass)/wall_insert_rho
        scaled_insert_kappa = (
            experimental_kappa(temperature)
            * puma_kappa(mass_loss_frac)
            / puma_kappa(0))
        return (scaled_insert_kappa)

    def _get_wall_kappa_inert(mass, temperature):
        return wall_insert_kappa * wall_mask

    def _get_wall_effective_surface_area_fiber(mass):
        mass_loss_frac = (wall_insert_rho - mass)/wall_insert_rho
        return puma_effective_surface_area(mass_loss_frac)

    def _mass_loss_rate_fiber(mass, ox_mass, temperature, eff_surf_area):
        actx = mass.array_context
        alpha = (
            (0.00143+0.01*actx.np.exp(-1450.0/temperature))
            / (1.0+0.0002*actx.np.exp(13000.0/temperature)))
        k = alpha*actx.np.sqrt(
            (univ_gas_const*temperature)/(2.0*np.pi*mw_o2))
        return (mw_co/mw_o2 + mw_o/mw_o2 - 1)*ox_mass*k*eff_surf_area

    # inert
    if wall_material == 0:
        wall_model = WallModel(
            heat_capacity=wall_insert_cp,
            thermal_conductivity_func=_get_wall_kappa_inert)
    # non-porous
    elif wall_material == 1:
        wall_model = WallModel(
            heat_capacity=wall_insert_cp,
            thermal_conductivity_func=_get_wall_kappa_fiber,
            effective_surface_area_func=_get_wall_effective_surface_area_fiber,
            mass_loss_func=_mass_loss_rate_fiber,
            oxygen_diffusivity=wall_insert_ox_diff)
    # porous
    elif wall_material == 2:
        wall_model = WallModel(
            heat_capacity=wall_insert_cp,
            thermal_conductivity_func=_get_wall_kappa_fiber,
            effective_surface_area_func=_get_wall_effective_surface_area_fiber,
            mass_loss_func=_mass_loss_rate_fiber,
            oxygen_diffusivity=wall_insert_ox_diff)

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "step walltime: {value:6g} s, ")
            #("t_log.max", "log walltime: {value:6g} s")
        ])

        try:
            logmgr.add_watches(["memory_usage_python.max"])
            #logmgr.add_watches(["memory_usage_python.max", "memory: {value}"])
        except KeyError:
            pass

        try:
            logmgr.add_watches(["memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    wall_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_wall)
    # fluid_oi_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid,)
    # wall_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_wall)

    #    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_write_status(cv, dv, wall_temperature, dt, cfl_fluid, cfl_wall):
        status_msg = (f"-------- dt = {dt:1.3e},"
                      f" cfl_fluid = {cfl_fluid:1.8f}"
                      f" cfl_wall = {cfl_wall:1.8f}")

        pmin = vol_min(dd_vol_fluid, dv.pressure)
        pmax = vol_max(dd_vol_fluid, dv.pressure)
        tmin = vol_min(dd_vol_fluid, dv.temperature)
        tmax = vol_max(dd_vol_fluid, dv.temperature)
        twmin = vol_min(dd_vol_wall, wall_temperature)
        twmax = vol_max(dd_vol_wall, wall_temperature)

        from pytools.obj_array import obj_array_vectorize
        y_min = obj_array_vectorize(lambda x: vol_min(dd_vol_fluid, x),
                                    cv.species_mass_fractions)
        y_max = obj_array_vectorize(lambda x: vol_max(dd_vol_fluid, x),
                                    cv.species_mass_fractions)

        dv_status_msg = (
            f"\n-------- P (min, max) (Pa) = ({pmin:1.9e}, {pmax:1.9e})")
        dv_status_msg += (
            f"\n-------- T_fluid (min, max) (K)  = ({tmin:7g}, {tmax:7g})")
        dv_status_msg += (
            f"\n-------- T_wall (min, max) (K)  = ({twmin:7g}, {twmax:7g})")

        if nspecies > 2:
            # check the temperature convergence
            # a single call to get_temperature_update is like taking an additional
            # Newton iteration and gives us a residual
            temp_resid = get_temperature_update_compiled(
                cv, dv.temperature)/dv.temperature
            temp_err_min = vol_min(dd_vol_fluid, temp_resid)
            temp_err_max = vol_max(dd_vol_fluid, temp_resid)
            dv_status_msg += (
                f"\n-------- T_resid (min, max) = "
                f"({temp_err_min:1.5e}, {temp_err_max:1.5e})")

        for i in range(nspecies):
            dv_status_msg += (
                f"\n-------- y_{species_names[i]} (min, max) = "
                f"({y_min[i]:1.3e}, {y_max[i]:1.3e})")
        status_msg += dv_status_msg
        status_msg += "\n"

        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, fluid_state, wv, cv_limited, wall_kappa,
                     wall_temperature, ts_field_fluid, ts_field_wall,
                     dump_number):

        if rank == 0:
            print(f"******** Writing Visualization File {dump_number}"
                  f" at step {step},"
                  f" sim time {t:1.6e} s ********")

        cv = fluid_state.cv
        dv = fluid_state.dv
        mu = fluid_state.viscosity

        # basic viz quantities, things here are difficult (or impossible) to compute
        # in post-processing
        fluid_viz_fields = [("cv", cv),
                            #("cv_limited", cv_limited),
                            ("dv", dv),
                            ("dt" if constant_cfl else "cfl", ts_field_fluid)]
        wall_viz_fields = [
            ("wv", wv),
            ("wall_kappa", wall_kappa),
            ("wall_temperature", wall_temperature),
            ("dt" if constant_cfl else "cfl", ts_field_wall)
        ]

        # extra viz quantities, things here are often used for post-processing
        if viz_level > 0:
            mach = fluid_state.speed / dv.speed_of_sound
            fluid_viz_ext = [("mach", mach),
                             ("velocity", cv.velocity)]
            fluid_viz_fields.extend(fluid_viz_ext)

            # species mass fractions
            fluid_viz_fields.extend(
                ("Y_"+species_names[i], cv.species_mass_fractions[i])
                for i in range(nspecies))

            #fluid_viz_fields.extend(
                #("Y_limited_"+species_names[i], cv_limited.species_mass_fractions[i])
                #for i in range(nspecies))

            if nspecies > 2:
                temp_resid = get_temperature_update_compiled(
                    cv_limited, dv.temperature)/dv.temperature
                production_rates = compute_production_rates(fluid_state.cv,
                                                            fluid_state.temperature)
                fluid_viz_ext = [("temp_resid", temp_resid),
                                 ("production_rates", production_rates)]
                fluid_viz_fields.extend(fluid_viz_ext)

            if use_av:
                fluid_viz_ext = [("mu", mu)]
                fluid_viz_fields.extend(fluid_viz_ext)

            if nparts > 1:
                fluid_viz_ext = [("rank", rank)]
                fluid_viz_fields.extend(fluid_viz_ext)

        # additional viz quantities, add in some non-dimensional numbers
        if viz_level > 1:
            sound_speed = fluid_state.speed_of_sound
            cell_Re = (cv_limited.mass*cv_limited.speed*char_length /
                fluid_state.viscosity)
            cp = gas_model.eos.heat_capacity_cp(cv_limited, fluid_state.temperature)
            alpha_heat = fluid_state.thermal_conductivity/cp/fluid_state.viscosity
            cell_Pe_heat = char_length*cv_limited.speed/alpha_heat
            from mirgecom.viscous import get_local_max_species_diffusivity
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )
            cell_Pe_mass = char_length*cv_limited.speed/d_alpha_max
            # these are useful if our transport properties
            # are not constant on the mesh
            # prandtl
            # schmidt_number
            # damkohler_number

            viz_ext = [("Re", cell_Re),
                       ("Pe_mass", cell_Pe_mass),
                       ("Pe_heat", cell_Pe_heat),
                       ("c", sound_speed)]
            fluid_viz_fields.extend(viz_ext)

            cell_alpha = wall_model.thermal_diffusivity(
                wv.mass, wall_temperature, wall_kappa)

            viz_ext = [
                       ("alpha", cell_alpha)]
            wall_viz_fields.extend(viz_ext)

        # debbuging viz quantities, things here are used for diagnosing run issues
        if viz_level > 2:
            """
            from mirgecom.fluid import (
                velocity_gradient,
                species_mass_fraction_gradient
            )
            ns_rhs, grad_cv, grad_t = \
                ns_operator(dcoll, state=fluid_state, time=t,
                            boundaries=boundaries, gas_model=gas_model,
                            return_gradients=True)
            grad_v = velocity_gradient(cv, grad_cv)
            grad_y = species_mass_fraction_gradient(cv, grad_cv)
            """

            grad_temperature = grad_t_operator(
                t, fluid_state, wall_kappa, wall_temperature)
            fluid_grad_temperature = grad_temperature[0]
            wall_grad_temperature = grad_temperature[1]
            grad_cv = grad_cv_operator_compiled(fluid_state,
                                                time=t)
            from mirgecom.fluid import velocity_gradient
            grad_v = velocity_gradient(fluid_state.cv, grad_cv)

            from mirgecom.inviscid import inviscid_flux
            inv_flux = inviscid_flux(fluid_state)
            from mirgecom.viscous import viscous_flux
            visc_flux = viscous_flux(fluid_state, grad_cv, fluid_grad_temperature)

            viz_ext = [ ("grad_temperature", fluid_grad_temperature),
                       ("grad_rho", grad_cv.mass),
                       ("grad_mom_x", grad_cv.momentum[0]),
                       ("grad_mom_y", grad_cv.momentum[1]),
                       ("grad_v_x", grad_v[0]),
                       ("grad_v_y", grad_v[1]),
                       ("grad_e", grad_cv.energy),
                       ("inv_flux_rho", inv_flux.mass),
                       ("inv_flux_rhoE", inv_flux.energy),
                       ("inv_flux_mom_x", inv_flux.momentum[0]),
                       ("inv_flux_mom_y", inv_flux.momentum[1]),
                       ("visc_flux_rho", visc_flux.mass),
                       ("visc_flux_rhoE", visc_flux.energy),
                       ("visc_flux_mom_x", visc_flux.momentum[0]),
                       ("visc_flux_mom_y", visc_flux.momentum[1])]
            if dim == 3:
                viz_ext.extend(("grad_v_z", grad_cv.velocity[2]))

            """
            viz_ext.extend(("grad_Y_"+species_names[i], grad_y[i])
                           for i in range(nspecies))
            """
            fluid_viz_fields.extend(viz_ext)

            viz_ext = [("grad_temperature", wall_grad_temperature)]
            wall_viz_fields.extend(viz_ext)

        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=dump_number, t=t,
            overwrite=True, comm=comm)
        write_visfile(
            dcoll, wall_viz_fields, wall_visualizer,
            vizname=vizname+"-wall", step=dump_number, t=t,
            overwrite=True, comm=comm)

        if rank == 0:
            print("******** Done Writing Visualization File ********\n")

    def my_write_restart(step, t, state):
        if rank == 0:
            print(f"******** Writing Restart File at step {step}, "
                  f"sim time {t:1.6e} s ********")

        cv, tseed, wv = state
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "temperature_seed": tseed,
                "nspecies": nspecies,
                "wv": wv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

        if rank == 0:
            print("******** Done Writing Restart File ********\n")

    def my_health_check(fluid_state):
        # FIXME: Add health check for wall temperature?
        health_error = False
        cv = fluid_state.cv
        dv = fluid_state.dv

        if check_naninf_local(dcoll, dd_vol_fluid, dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_range_check(dd_vol_fluid, dv.pressure,
                              health_pres_min, health_pres_max):
            health_error = True
            p_min = vol_min(dd_vol_fluid, dv.pressure)
            p_max = vol_max(dd_vol_fluid, dv.pressure)
            logger.info(f"Pressure range violation: "
                        f"Simulation Range ({p_min=}, {p_max=}) "
                        f"Specified Limits ({health_pres_min=}, {health_pres_max=})")

        if global_range_check(dd_vol_fluid, dv.temperature,
                              health_temp_min, health_temp_max):
            health_error = True
            t_min = vol_min(dd_vol_fluid, dv.temperature)
            t_max = vol_max(dd_vol_fluid, dv.temperature)
            logger.info(f"Temperature range violation: "
                        f"Simulation Range ({t_min=}, {t_max=}) "
                        f"Specified Limits ({health_temp_min=}, {health_temp_max=})")

        for i in range(nspecies):
            if global_range_check(dd_vol_fluid, cv.species_mass_fractions[i],
                                  health_mass_frac_min, health_mass_frac_max):
                health_error = True
                y_min = vol_min(dd_vol_fluid, cv.species_mass_fractions[i])
                y_max = vol_max(dd_vol_fluid, cv.species_mass_fractions[i])
                logger.info(f"Species mass fraction range violation. "
                            f"{species_names[i]}: ({y_min=}, {y_max=})")

        if nspecies > 2:
            # check the temperature convergence
            # a single call to get_temperature_update is like taking an additional
            # Newton iteration and gives us a residual
            temp_resid = get_temperature_update_compiled(
                cv, dv.temperature)/dv.temperature
            temp_err = vol_max(dd_vol_fluid, temp_resid)
            if temp_err > pyro_temp_tol:
                health_error = True
                logger.info(f"Temperature is not converged "
                            f"{temp_err=} > {pyro_temp_tol}.")

        return health_error

    def my_get_viscous_timestep(dcoll, fluid_state):
        """Routine returns the the node-local maximum stable viscous timestep.

        Parameters
        ----------
        dcoll: grudge.eager.EagerDGDiscretization
            the discretization to use
        fluid_state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """
        nu = 0
        d_alpha_max = 0

        if fluid_state.is_viscous:
            from mirgecom.viscous import get_local_max_species_diffusivity
            nu = fluid_state.viscosity/fluid_state.mass_density
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )

        return (
            char_length / (fluid_state.wavespeed
            + ((nu + d_alpha_max) / char_length))
        )

    def my_get_wall_timestep(dcoll, wv, wall_kappa, wall_temperature):
        """Routine returns the the node-local maximum stable thermal timestep.

        Parameters
        ----------
        dcoll: grudge.eager.EagerDGDiscretization
            the discretization to use

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """

        return (
            char_length_wall*char_length_wall
            / (
                wall_time_scale
                * actx.np.maximum(
                    wall_model.thermal_diffusivity(
                        wv.mass, wall_temperature, wall_kappa),
                    wall_model.oxygen_diffusivity)))

    def _my_get_timestep_wall(
            dcoll, wv, wall_kappa, wall_temperature, t, dt, cfl, t_final,
            constant_cfl=False, wall_dd=DD_VOLUME_ALL):
        """Return the maximum stable timestep for a typical heat transfer simulation.

        This routine returns *dt*, the users defined constant timestep, or *max_dt*,
        the maximum domain-wide stability-limited timestep for a fluid simulation.

        .. important::
            This routine calls the collective: :func:`~grudge.op.nodal_min` on the
            inside which makes it domain-wide regardless of parallel domain
            decomposition. Thus this routine must be called *collectively*
            (i.e. by all ranks).

        Two modes are supported:
            - Constant DT mode: returns the minimum of (t_final-t, dt)
            - Constant CFL mode: returns (cfl * max_dt)

        Parameters
        ----------
        dcoll
            Grudge discretization or discretization collection?
        t: float
            Current time
        t_final: float
            Final time
        dt: float
            The current timestep
        cfl: float
            The current CFL number
        constant_cfl: bool
            True if running constant CFL mode

        Returns
        -------
        float
            The dt (contant cfl) or cfl (constant dt) at every point in the mesh
        float
            The minimum stable cfl based on conductive heat transfer
        float
            The maximum stable DT based on conductive heat transfer
        """
        actx = wall_kappa.array_context
        mydt = dt
        if constant_cfl:
            ts_field = cfl*my_get_wall_timestep(
                dcoll=dcoll, wv=wv, wall_kappa=wall_kappa,
                wall_temperature=wall_temperature)
            mydt = actx.to_numpy(
                op.nodal_min(
                    dcoll, wall_dd, ts_field, initial=np.inf))[()]
        else:
            ts_field = mydt/my_get_wall_timestep(
                dcoll=dcoll, wv=wv, wall_kappa=wall_kappa,
                wall_temperature=wall_temperature)
            cfl = actx.to_numpy(
                op.nodal_max(
                    dcoll, wall_dd, ts_field, initial=0.))[()]

        return ts_field, cfl, mydt

    #my_get_timestep = actx.compile(_my_get_timestep)
    my_get_timestep_wall = _my_get_timestep_wall

    def _my_get_timestep(
            dcoll, fluid_state, t, dt, cfl, t_final, constant_cfl=False,
            fluid_dd=DD_VOLUME_ALL):
        """Return the maximum stable timestep for a typical fluid simulation.

        This routine returns *dt*, the users defined constant timestep, or *max_dt*,
        the maximum domain-wide stability-limited timestep for a fluid simulation.

        .. important::
            This routine calls the collective: :func:`~grudge.op.nodal_min` on the
            inside which makes it domain-wide regardless of parallel domain
            decomposition. Thus this routine must be called *collectively*
            (i.e. by all ranks).

        Two modes are supported:
            - Constant DT mode: returns the minimum of (t_final-t, dt)
            - Constant CFL mode: returns (cfl * max_dt)

        Parameters
        ----------
        dcoll
            Grudge discretization or discretization collection?
        fluid_state: :class:`~mirgecom.gas_model.FluidState`
            The full fluid conserved and thermal state
        t: float
            Current time
        t_final: float
            Final time
        dt: float
            The current timestep
        cfl: float
            The current CFL number
        constant_cfl: bool
            True if running constant CFL mode

        Returns
        -------
        float
            The dt (contant cfl) or cfl (constant dt) at every point in the mesh
        float
            The minimum stable cfl based on a viscous fluid.
        float
            The maximum stable DT based on a viscous fluid.
        """
        mydt = dt
        if constant_cfl:
            ts_field = cfl*my_get_viscous_timestep(
                dcoll=dcoll, fluid_state=fluid_state)
            mydt = fluid_state.array_context.to_numpy(op.nodal_min(
                dcoll, fluid_dd, ts_field, initial=np.inf))[()]
        else:
            ts_field = mydt/my_get_viscous_timestep(
                dcoll=dcoll, fluid_state=fluid_state)
            cfl = fluid_state.array_context.to_numpy(op.nodal_max(
                dcoll, fluid_dd, ts_field, initial=0.))[()]

        return ts_field, cfl, mydt

    #my_get_timestep = actx.compile(_my_get_timestep)
    my_get_timestep = _my_get_timestep

    def limit_species(fluid_state):
        cv = fluid_state.cv
        temperature = fluid_state.temperature
        pressure = fluid_state.pressure

        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll=dcoll, dd=dd_vol_fluid,
                                     field=cv.species_mass_fractions[i],
                                     mmin=0.0, mmax=1.0, modify_average=True)
            for i in range(nspecies)
        ])

        # limit the sum to 1.0
        aux = cv.mass*0.0
        for i in range(nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        kin_energy = 0.5*np.dot(cv.velocity, cv.velocity)

        mass_lim = eos.get_density(pressure=pressure, temperature=temperature,
                                   species_mass_fractions=spec_lim)

        energy_lim = mass_lim*(
            gas_model.eos.get_internal_energy(temperature,
                                              species_mass_fractions=spec_lim)
            + kin_energy
        )

        mom_lim = mass_lim*cv.velocity

        cv_limited = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                                    momentum=mom_lim,
                                    species_mass=mass_lim*spec_lim)

        return cv_limited

    limit_species_compiled = actx.compile(limit_species)

    def my_pre_step(step, t, dt, state):

        # Filter *first* because this will be most straightfwd to
        # understand and move. For this to work, this routine
        # must pass back the filtered CV in the state.
        if check_step(step=step, interval=soln_nfilter):
            cv, tseed, wv = state
            cv = filter_cv_compiled(cv)
            state = make_obj_array([cv, tseed, wv])

        cv, tseed, wv = state
        fluid_state = create_fluid_state(cv=cv,
                                         temperature_seed=tseed,
                                         smoothness=no_smoothness)
        wdv = create_wall_dependent_vars_compiled(wv)

        try:

            if logmgr:
                logmgr.tick_before()

            # disable non-constant dt timestepping for now
            # re-enable when we're ready

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)
            do_limit = check_step(step=step, interval=nlimit)
            next_dump_number = step

            # if the species limiter is on, compute the limited cv
            cv_limited = cv
            if use_species_limiter > 0:
                cv_limited = limit_species_compiled(fluid_state)

            # update the cv to be the limited case
            if use_species_limiter and do_limit:
                cv_resid = cv_limited - cv
                cv = cv_limited

                fluid_state = update_fluid_state_compiled(cv=cv,
                                                          dv=fluid_state.dv,
                                                          tv=fluid_state.tv)

            state = make_obj_array([cv, fluid_state.temperature, wv])

            if any([do_viz, do_restart, do_health, do_status]):
                # compute the limited cv so we can viz what the rhs sees

                if use_av:
                    # limited cv here to compute smoothness
                    fluid_state = update_fluid_state_compiled(
                        cv=cv_limited, dv=fluid_state.dv, tv=fluid_state.tv)

                    # recompute the dv to have the correct smoothness
                    if do_viz:
                        # use the divergence to compute the smoothness field
                        grad_cv = grad_cv_operator_compiled(fluid_state,
                                                            time=t)
                        # limited cv here to compute smoothness
                        smoothness = compute_smoothness_compiled(
                            cv=cv_limited, dv=fluid_state.dv,
                            grad_cv=grad_cv)

                        # unlimited cv here as that is what gets written
                        dv_new = update_dv_compiled(
                            cv=cv, temperature=fluid_state.temperature,
                            smoothness=smoothness)
                        tv_new = update_tv_compiled(cv=cv, dv=dv_new)
                        fluid_state = update_fluid_state_compiled(
                            cv=cv, dv=dv_new, tv=tv_new)

                # pass through, removes a bunch of tagging to avoid recomplie
                wv = get_wv(wv)

                if not force_eval:
                    fluid_state = force_evaluation(actx, fluid_state)
                    wv = force_evaluation(actx, wv)
                    cv_limited = force_evaluation(actx, cv_limited)

                dv = fluid_state.dv

                ts_field_fluid, cfl_fluid, dt_fluid = my_get_timestep(
                    dcoll=dcoll, fluid_state=fluid_state,
                    t=t, dt=dt, cfl=current_cfl, t_final=t_final,
                    constant_cfl=constant_cfl, fluid_dd=dd_vol_fluid)

                ts_field_wall, cfl_wall, dt_wall = my_get_timestep_wall(
                    dcoll=dcoll, wv=wv, wall_kappa=wdv.thermal_conductivity,
                    wall_temperature=wdv.temperature, t=t, dt=dt,
                    cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
                    wall_dd=dd_vol_wall)

            # update the cv to be the limited case
            # save the difference in cv_limited for viz
            if use_species_limiter and do_limit:
                cv_limited = force_evaluation(actx, cv_resid)

            state = make_obj_array([cv, fluid_state.temperature, wv])

            if do_health:
                health_errors = global_reduce(my_health_check(fluid_state), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.warning("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_status:
                my_write_status(dt=dt, cfl_fluid=cfl_fluid, cfl_wall=cfl_wall,
                                cv=cv, dv=dv, wall_temperature=wdv.temperature)

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(
                    step=step, t=t, fluid_state=fluid_state,
                    wv=wv, wall_kappa=wdv.thermal_conductivity,
                    wall_temperature=wdv.temperature, ts_field_fluid=ts_field_fluid,
                    ts_field_wall=ts_field_wall,
                    dump_number=next_dump_number, cv_limited=cv_limited)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")

            dump_number = step

            if use_av:
                # limited cv here to compute smoothness
                fluid_state = update_fluid_state_compiled(
                    cv=cv_limited, dv=fluid_state.dv, tv=fluid_state.tv)

                # use the divergence to compute the smoothness field
                grad_cv = grad_cv_operator_compiled(fluid_state,
                                                    time=t)
                # limited cv here to compute smoothness
                smoothness = compute_smoothness_compiled(
                    cv=cv_limited, dv=fluid_state.dv,
                    grad_cv=grad_cv)

                # unlimited cv here as that is what gets written
                dv_new = update_dv_compiled(
                    cv=cv, temperature=fluid_state.temperature,
                    smoothness=smoothness)
                tv_new = update_tv_compiled(cv=cv, dv=dv_new)
                fluid_state = update_fluid_state_compiled(
                    cv=cv, dv=dv_new, tv=tv_new)

                # pass through, removes a bunch of tagging to avoid recomplie
                wv = get_wv(wv)

                if not force_eval:
                    fluid_state = force_evaluation(actx, fluid_state)
                    wv = force_evaluation(actx, wv)
                    cv_limited = force_evaluation(actx, cv_limited)

                dv = fluid_state.dv

                ts_field_fluid, cfl_fluid, dt_fluid = my_get_timestep(
                    dcoll=dcoll, fluid_state=fluid_state,
                    t=t, dt=dt, cfl=current_cfl, t_final=t_final,
                    constant_cfl=constant_cfl, fluid_dd=dd_vol_fluid)

                ts_field_wall, cfl_wall, dt_wall = my_get_timestep_wall(
                    dcoll=dcoll, wv=wv, wall_kappa=wdv.thermal_conductivity,
                    wall_temperature=wdv.temperature, t=t, dt=dt,
                    cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
                    wall_dd=dd_vol_wall)

            my_write_viz(
                step=step, t=t, fluid_state=fluid_state,
                wv=wv, wall_kappa=wdv.thermal_conductivity,
                wall_temperature=wdv.temperature, ts_field_fluid=ts_field_fluid,
                ts_field_wall=ts_field_wall,
                dump_number=dump_number, cv_limited=cv_limited)
            my_write_restart(step=step, t=t, state=state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    flux_beta = 1
    #flux_alpha = 0.5
    #flux_beta = 0.

    from mirgecom.viscous import viscous_flux
    from mirgecom.flux import num_flux_central

    def _num_flux_dissipative(u_minus, u_plus, alpha, beta):
        u_minus_normal = u_minus@normal
        u_plus_normal = u_plus@normal
        #return num_flux_central(u_minus, u_plus) + beta*(u_plus - u_minus)/2
        return (num_flux_central(u_minus, u_plus) -
                beta*np.dot(normal, u_plus_normal + u_minus_normal)/2)

    def _num_flux_ldg(u_minus, u_plus, alpha, beta):
        u_minus_normal = u_minus@normal
        u_plus_normal = u_plus@normal
        #return num_flux_central(u_minus, u_plus) + beta*(u_plus - u_minus)/2
        return (num_flux_central(u_minus, u_plus) -
                beta*np.dot(normal, u_plus_normal + u_minus_normal)/2)

    def _viscous_facial_flux_dissipative(dcoll, state_pair, grad_cv_pair,
                                         grad_t_pair, beta=0., gas_model=None):
        actx = state_pair.int.array_context
        normal = actx.thaw(dcoll.normal(state_pair.dd))

        f_int = viscous_flux(state_pair.int, grad_cv_pair.int,
                             grad_t_pair.int)
        f_ext = viscous_flux(state_pair.ext, grad_cv_pair.ext,
                             grad_t_pair.ext)

        vel_int = state_pair.int.velocity
        vel_ext = state_pair.ext.velocity

        temp_int = state_pair.int.temperature
        temp_ext = state_pair.ext.temperature

        num_visc_flux = num_flux_central(f_int, f_ext)

        #vel_penalty = beta*(vel_ext*normal - vel_int*normal)/2.
        #temp_penalty = beta*(temp_ext*normal - temp_int*normal)/2.

        #vel_penalty = -beta*(vel_ext*normal - vel_int*normal)
        vel_penalty = -beta*(np.dot(vel_int, normal) -
                            np.dot(vel_ext, normal))*normal
        temp_penalty = -beta*(temp_int - temp_ext)*normal

        visc_flux_vel = num_visc_flux.momentum - vel_penalty
        visc_flux_temp = num_visc_flux.energy - temp_penalty

        num_visc_flux_penalized = make_conserved(
            dim=state_pair.int.dim,
            mass=num_visc_flux.mass,
            energy=visc_flux_temp,
            momentum=visc_flux_vel,
            species_mass=num_visc_flux.species_mass)

        return num_visc_flux_penalized@normal

        #f_int_normal = f_int@normal
        #f_ext_normal = f_ext@normal

        #return (num_flux_central(f_int, f_ext) +
                #beta*np.outer(normal, f_int_normal + f_ext_normal)/2)@normal

        #return _num_flux_dissipative(f_int, f_ext, beta=beta)@normal

        #return (num_flux_central(f_int, f_ext) -
                #beta*normal*(state_pair.int - state_pair.ext)/2)@normal

    #grad_num_flux_func = partial(_num_flux_dissipative, beta=flux_beta)
    #viscous_num_flux_func = partial(_viscous_facial_flux_dissipative,
                                    #beta=flux_beta)
    grad_num_flux_func = num_flux_central
    viscous_num_flux_func = num_flux_central

    def my_rhs(t, state):
        cv, tseed, wv = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed,
                                       smoothness=no_smoothness)

        if use_av:
            # use the divergence to compute the smoothness field
            grad_fluid_cv = grad_cv_operator(
                dcoll, gas_model, fluid_boundaries, fluid_state,
                dd=dd_vol_fluid,
                time=t, quadrature_tag=quadrature_tag)
            smoothness = compute_smoothness(cv=cv, dv=fluid_state.dv,
                                            grad_cv=grad_fluid_cv)

            dv_new = update_dv(cv=cv, temperature=fluid_state.temperature,
                               smoothness=smoothness)
            tv_new = update_tv(cv=cv, dv=dv_new)
            fluid_state = update_fluid_state(cv=cv, dv=dv_new, tv=tv_new)

        # update wall model
        wdv = wall_model.dependent_vars(wv)

        # Temperature seed RHS (keep tseed updated)
        tseed_rhs = fluid_state.temperature - tseed

        """
        # Steps common to NS and AV (and wall model needs grad(temperature))
        operator_fluid_states = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, boundaries, quadrature_tag)

        grad_fluid_cv = grad_cv_operator(
            dcoll, gas_model, boundaries, fluid_state,
            quadrature_tag=quadrature_tag,
            operator_states_quad=operator_fluid_states)
        """

        ns_use_av = False
        ns_rhs, wall_energy_rhs = coupled_ns_heat_operator(
            dcoll=dcoll,
            gas_model=gas_model,
            fluid_dd=dd_vol_fluid, wall_dd=dd_vol_wall,
            fluid_boundaries=fluid_boundaries,
            wall_boundaries=wall_boundaries,
            inviscid_numerical_flux_func=inviscid_numerical_flux_func,
            #viscous_numerical_flux_func=viscous_num_flux_func,
            fluid_gradient_numerical_flux_func=grad_num_flux_func,
            fluid_state=fluid_state,
            wall_kappa=wdv.thermal_conductivity,
            wall_temperature=wdv.temperature,
            time=t,
            wall_penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag)

        sponge_rhs = 0*cv
        if use_sponge:
            sponge_rhs = _sponge_source(cv=cv)

        fluid_rhs = ns_rhs + sponge_rhs

        #wall_mass_rhs = -wall_model.mass_loss_rate(wv)
        # wall mass loss
        wall_mass_rhs = 0.*wv.mass
        if use_wall_mass:
            wall_mass_rhs = -wall_model.mass_loss_rate(
                mass=wv.mass, ox_mass=wv.ox_mass,
                temperature=wdv.temperature)

        # wall oxygen diffusion
        wall_ox_mass_rhs = 0.*wv.ox_mass
        if use_wall_ox:
            if nspecies > 2:
                fluid_ox_mass = cv.species_mass[i_ox]
            else:
                fluid_ox_mass = mf_o2*cv.species_mass[0]
            pairwise_ox = {
                (dd_vol_fluid, dd_vol_wall):
                    (fluid_ox_mass, wv.ox_mass)}
            pairwise_ox_tpairs = inter_volume_trace_pairs(
                dcoll, pairwise_ox, comm_tag=_OxCommTag)
            ox_tpairs = pairwise_ox_tpairs[dd_vol_fluid, dd_vol_wall]

            wall_ox_boundaries = {
                dd_vol_wall.trace("wall_farfield").domain_tag:
                    DirichletDiffusionBoundary(0)}
            wall_ox_boundaries.update({
                tpair.dd.domain_tag: DirichletDiffusionBoundary(
                    op.project(dcoll, tpair.dd,
                               tpair.dd.with_discr_tag(quadrature_tag), tpair.ext))
                for tpair in ox_tpairs})

            wall_ox_mass_rhs = diffusion_operator(
                dcoll, wall_model.oxygen_diffusivity, wall_ox_boundaries, wv.ox_mass,
                penalty_amount=wall_penalty_amount,
                quadrature_tag=quadrature_tag, dd=dd_vol_wall,
                comm_tag=_WallOxDiffCommTag)

            # Solve a diffusion equation in the fluid too just to ensure all MPI
            # sends/recvs from inter_volume_trace_pairs are in DAG
            # FIXME: this is dumb
            reverse_ox_tpairs = pairwise_ox_tpairs[dd_vol_wall, dd_vol_fluid]

            fluid_ox_boundaries = {
                bdtag: DirichletDiffusionBoundary(0)
                for bdtag in fluid_boundaries}
            fluid_ox_boundaries.update({
                tpair.dd.domain_tag: DirichletDiffusionBoundary(
                    op.project(dcoll, tpair.dd,
                               tpair.dd.with_discr_tag(quadrature_tag), tpair.ext))
                for tpair in reverse_ox_tpairs})

            fluid_dummy_ox_mass_rhs = diffusion_operator(
                dcoll, 0, fluid_ox_boundaries, fluid_ox_mass,
                quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
                comm_tag=_FluidOxDiffCommTag)

            fluid_rhs = fluid_rhs + 0*fluid_dummy_ox_mass_rhs

        # Use a spectral filter on the RHS
        if use_rhs_filter:
            fluid_rhs = filter_rhs(fluid_rhs)

        wall_rhs = wall_time_scale * WallVars(
            mass=wall_mass_rhs,
            energy=wall_energy_rhs,
            ox_mass=wall_ox_mass_rhs)

        return make_obj_array([fluid_rhs, tseed_rhs, wall_rhs])

    """
    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)
    """

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt,
                      t=current_t, t_final=t_final,
                      force_eval=force_eval,
                      state=stepper_state)
    current_cv, tseed, current_wv = stepper_state
    current_fluid_state = create_fluid_state(current_cv, tseed)
    current_wdv = create_wall_dependent_vars_compiled(current_wv)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    limit_species_rhs = 0*current_fluid_state.cv
    current_cv_limited = current_cv
    if use_species_limiter > 0:
        fluid_state = create_fluid_state(cv=current_fluid_state.cv,
                                         temperature_seed=tseed,
                                         smoothness=no_smoothness)
        current_cv_limited, limit_species_rhs = limit_species_compiled(
            fluid_state)

    if use_av == 0:
        current_fluid_state = create_fluid_state(cv=current_cv,
                                                 smoothness=no_smoothness,
                                                 temperature_seed=tseed)
    elif use_av:
        current_fluid_state = create_fluid_state(cv=current_cv_limited,
                                                 temperature_seed=tseed,
                                                 smoothness=no_smoothness)

        # use the divergence to compute the smoothness field
        current_grad_cv = grad_cv_operator_compiled(
            fluid_state=current_fluid_state, time=current_t)
        smoothness = compute_smoothness_compiled(
            cv=current_cv, dv=current_fluid_state.dv, grad_cv=current_grad_cv)

        current_fluid_state = create_fluid_state(cv=current_cv,
                                           temperature_seed=tseed,
                                           smoothness=smoothness)

    final_dv = current_fluid_state.dv
    ts_field_fluid, cfl, dt = my_get_timestep(dcoll=dcoll,
        fluid_state=current_fluid_state,
        t=current_t, dt=current_dt, cfl=current_cfl,
        t_final=t_final, constant_cfl=constant_cfl, fluid_dd=dd_vol_fluid)
    ts_field_wall, cfl_wall, dt_wall = my_get_timestep_wall(dcoll=dcoll,
        wv=current_wv, wall_kappa=current_wdv.thermal_conductivity,
        wall_temperature=current_wdv.temperature, t=current_t, dt=current_dt,
        cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
        wall_dd=dd_vol_wall)
    my_write_status(dt=dt, cfl_fluid=cfl, cfl_wall=cfl_wall,
                    dv=final_dv, cv=current_cv,
                    wall_temperature=current_wdv.temperature)

    dump_number = current_step

    my_write_viz(
        step=current_step, t=current_t, fluid_state=current_fluid_state,
        cv_limited=current_cv_limited,
        wv=current_wv, wall_kappa=current_wdv.thermal_conductivity,
        wall_temperature=current_wdv.temperature,
        ts_field_fluid=ts_field_fluid,
        ts_field_wall=ts_field_wall,
        dump_number=dump_number)
    my_write_restart(step=current_step, t=current_t, state=stepper_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    #root_logger = logging.getLogger()

    #logging.debug("A DEBUG message")
    #logging.info("An INFO message")
    #logging.warning("A WARNING message")
    #logging.error("An ERROR message")
    #logging.critical("A CRITICAL message")

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-t", "--target_file", type=ascii, dest="target_file",
                        nargs="?", action="store", help="simulation target file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")

    args = parser.parse_args()

    # for writing output
    casename = "prediction"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")
    lazy = args.lazy
    if args.profile:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    target_filename = None
    if args.target_file:
        target_filename = (args.target_file).replace("'", "")
        print(f"Target file specified: {target_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Using user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, target_filename=target_filename,
         user_input_file=input_file,
         use_profiling=args.profile, use_logmgr=args.log,
         use_overintegration=args.overintegration, lazy=lazy,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
