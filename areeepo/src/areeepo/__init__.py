################################################################################
# Copyright (c) 2026 Patrick Hirling (patrick.hirling@epfl.ch)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################

import numpy as np
import astropy.cosmology as apco
from .io import *
from .utils import *
from .visual import *

# ///////////////////////////////////////////////////////////////
# 'areeepo' python module
#
# A set of routines to handle HDF5 snapshots of AREPO
# simulations. Designed to work with the SGChem chemistry module,
# and SF_ECOGAL star formation module.
#
# Contains functions to interpolate a gas distribution on a 3D
# grid, to make slices and projections and visualise the output
# of a simulation.
# ///////////////////////////////////////////////////////////////

# ============================================================= #
# FUNCTIONS TO COMPUTE DERIVED QUANTITIES FROM SNAPSHOTS
# ============================================================= #

def get_time_Myr(f, cosmology : apco.Cosmology = apco.Planck15, z_init = np.inf):
    """
    Get current time in Myr, counted since `z_init`
    
    Parameters
    ----------
    f : h5py.File
        Snapshot file
    cosmology : astropy.cosmology.Cosmology
        Cosmology to use to convert between time and redshift. Default: Planck (2015)
    z_init : float
        Initial redshift from which to count. Default: z=inf (big bang)
    
    Returns
    -------
    time_Myr : float
        Time of snapshot in Myr
    """
    is_cosmo = is_cosmological(f)
    time = get_time(f)
    units = load_units(f)
    if is_cosmo:
        zred = 1./time - 1
        time_Myr = ((cosmology.age(zred) - cosmology.age(z_init)).to('Myr')).value
    else:
        time_Myr = time * units['UnitTime'] / Myr_cgs

    return time_Myr

def compute_mean_molecular_weight(f):
    """Compute the mean molecular weight of each gas cell in a snapshot.

    This function works for the SGChem chemistry module, or the default COOLING module
    used by e.g. IllustrisTNG.

    Parameters
    ----------
    f : h5py.File
        Snapshot file

    Returns
    -------
    mu : 3D array 
        Mean molecular weight of each gas cell (normalized to proton mass)
    """
    if 'SGCHEM' in f['Config'].attrs.keys():
        chemistry_network = int(get_attribute(f,'Config','CHEMISTRYNETWORK'))
        chemical_abundances = load_dataset_from_parttype(f,0,'ChemicalAbundances', remove_h_factors=False) # ChemicalAbundances has no h scaling

        # Variable metallicity
        if 'SGCHEM_VARIABLE_Z' in f['Config'].attrs.keys():
            ZAtom = load_dataset_from_parttype(f,0,'ElementAbundances', remove_h_factors=False)[:,3]
        else:
            ZAtom = float(f['Parameters'].attrs['ZAtom']) # Global metallicity, relative to solar
        
        if chemistry_network == 1:
            x_H2    = chemical_abundances[:,0]
            x_HII   = chemical_abundances[:,1]
            x_Dp    = chemical_abundances[:,2]
            x_HD    = chemical_abundances[:,3]
            x_Hep   = chemical_abundances[:,4]
            x_Hepp  = chemical_abundances[:,5]
        elif chemistry_network == 5 or chemistry_network == 10:
            x_H2    = chemical_abundances[:,0]
            x_HII   = chemical_abundances[:,1]
            x_CO    = chemical_abundances[:,2]
        else:
            raise ValueError("Unknown chemistry network: " + chemistry_network)
        
        x_HI = 1.0 - x_HII - 2*x_H2
        x_He = 0.1 * np.ones_like(x_HII) # TODO: make this more flexible

        # Compute hydrogen mass fraction & deduce hydrogen number density
        XH = _hydrogen_mass_fraction(ZAtom,0.1)

        # Mean molecular weight
        if chemistry_network == 5 or chemistry_network == 10:
            x_e = x_HII
            frac_sum = 1 + 0.1 - x_H2 + x_HII #x_HI + x_HII + x_H2 + x_e + x_He + x_Dp + x_HD + x_Hep + x_Hepp
        elif chemistry_network == 1:
            x_e = x_HII + x_Hep + 2*x_Hepp
            frac_sum = 1 + 0.079 - x_H2 + x_HII + x_Hep + 2*x_Hepp #x_HI + x_HII + x_H2 + x_e + x_He + x_CO
        
        mu = 1.0 / (XH * frac_sum)

        return mu
    
    elif 'COOLING' in f['Config'].attrs.keys() and not 'GRACKLE' in f['Config'].attrs.keys():
        xe = load_dataset_from_parttype(f,0,'ElectronAbundance', remove_h_factors=False)
        XH = 0.76
        
        mu = 4.0 / (1 + 3*XH + 4*XH*xe)

        return mu
        
    else:
        # TODO: add support for other chemistry modules
        raise ValueError("This run does not have chemistry")

def compute_temperature(f, gamma = 5.0/3.0):
    """Compute the physical temperature of each gas cell in a snapshot.

    Parameters
    ----------
    f : h5py.File
        Snapshot file
    gamma : float (optional)
        Adiabatic index. Default: 5/3

    Returns
    -------
    T : 3D array 
        Temperature in Kelvin of each gas cell
    """
    mu = compute_mean_molecular_weight(f)
    u = load_gas(f,'InternalEnergy')
    units = load_units(f)
    T = mP_cgs * mu * (gamma-1) * u * units['UnitEnergy']/units['UnitMass']/kB_cgs
    return T

def _hydrogen_mass_fraction(ZAtom,x_He=0.1,Zsolar = 0.0134):
    """Compute the hydrogen mass fraction X
    
    1 = X + Y + Z
    X: Hydrogen mass fraction
    Y: Helium mass fraction
    Z: "Metallicity"
    
    Arguments
    ---------
    Zmet : float
        Metallicity in terms of solar metallicity
    x_He : float, optional
        Helium abundance relative to total hydrogen atom abundance. Default: 0.1
    Zsolar : float, optional
        Solar metallicity. Default: 0.0134
    """
    return (1.0 - Zsolar*ZAtom) / (1.0 + 4*x_He)

# ============================================================= #
# EXPERIMENTAL
# ============================================================= #

def compute_species_ndens(f, i_species, convert_to_cgs = False):
    """ONLY FOR NETWORK 5 FOR NOW.
    i = 0: H2
    i = 1: HII
    i = 2: CO
    """
    chemical_abundances = load_dataset_from_parttype(f,0,'ChemicalAbundances', remove_h_factors=False) # ChemicalAbundances has no h scaling
    # TODO: Add option for variable Z
    ZAtom = float(f['Parameters'].attrs['ZAtom']) # Global metallicity, relative to solar
    x_species    = chemical_abundances[:,i_species]
    XH = _hydrogen_mass_fraction(ZAtom,0.1)
    ndens_H = XH * load_gas(f, 'Density', convert_to_cgs=1) / mP_cgs    # in CGS
    units = load_units(f)

    ndens_i = x_species * ndens_H
    if not convert_to_cgs:
        ndens_i *= units['UnitLength']**(-3)
    
    return ndens_i

# TODO: This method is a bit too "specific", probably it's best to just have a bunch
# of wrappers to call from notebooks
def make_image_gas(f, bins, quantities = 'density', extent = None, axis = 2, remove_h_factors = True):
    single_output = False
    if not isinstance(quantities,list):
        quantities = [quantities]
        single_output = True
    
    pos = load_dataset_from_parttype(f,0,'Coordinates',remove_h_factors=remove_h_factors) / 1000
    fields = []
    
    # Exclude low-resolution cells if its a zoom
    if is_zoom(f):
        hrmask = get_zoom_gas_mask(f)
    else:
        # If its not a zoom, we just take all particles
        hrmask = np.ones(pos.shape[0], dtype='bool')
    
    # We always project the density, as it is needed for other mass-weighted quantities
    dens = load_dataset_from_parttype(f,0,'Density',remove_h_factors=remove_h_factors)
    fields.append(dens[hrmask])

    if 'temperature' in quantities or 'u' in quantities:
        temp = compute_temperature(f)
        fields.append(temp[hrmask] * dens[hrmask]) # Mass-weighted temperature
    if 'internalenergy' in quantities:
        u = load_gas(f, 'InternalEnergy')
        fields.append(u[hrmask] * dens[hrmask]) # Energy density
    if 'H2' in quantities:
        nH2 = compute_species_ndens(f, 0, convert_to_cgs=1)
        fields.append(nH2[hrmask])
    if 'HII' in quantities:
        nHII = compute_species_ndens(f, 1, convert_to_cgs=1)
        fields.append(nHII[hrmask])
    if 'velocity' in quantities:
        vel = load_gas(f, 'Velocities')
        velnorm = np.sqrt( (vel**2).sum(axis=1) )
        fields.append(velnorm[hrmask] * dens[hrmask])
        
    # Only one call to projection method (so we only need to evaluate NN once for all fields)
    proj, bs, e2d = project_with_NN(pos[hrmask],fields,axis,bins,extent)
    
    res = {
        'binsizes' : bs,
        'extent' : e2d
    }
    # Only return projected density if explicitely asked for
    if 'density' in quantities:
        dens_proj = proj[0]
        res['density'] = dens_proj
    # Read other quantities
    count = 1
    if 'temperature' in quantities:
        temp_proj = proj[count] / proj[0]
        res['temperature'] = temp_proj
        count += 1
    if 'internalenergy' in quantities or 'u' in quantities:
        u_proj = proj[count] / proj[0]
        res['internalenergy'] = u_proj
        count += 1
    if 'H2' in quantities:
        H2_proj = proj[count]
        res['N_H2'] = H2_proj
        count += 1
    if 'HII' in quantities:
        HII_proj = proj[count]
        res['N_HII'] = HII_proj
        count += 1
    if 'velocity' in quantities:
        vel_proj = proj[count]
        res['velocity'] = vel_proj / proj[0]
        count += 1
        
    return res

def make_phase_plot_gas(f, bins, dens_unit = 'ncgs', remove_h_factors = True, log_extent = None):
    if dens_unit == 'ncgs':
        pdens = load_gas(f,'Density',remove_h_factors=remove_h_factors,convert_to_cgs=1) / mP_cgs
    elif dens_unit == 'cgs':
        pdens = load_gas(f,'Density',remove_h_factors=remove_h_factors,convert_to_cgs=1)
    elif dens_unit is None:
        pdens = load_gas(f,'Density',remove_h_factors=remove_h_factors,convert_to_cgs=0)

    pmass = load_gas(f,'Masses',remove_h_factors=remove_h_factors)
    ptemp = compute_temperature(f)

    # Exclude low-mass cells
    try:
        phrm = load_gas(f,'HighResGasMass')
    except KeyError:
        print("Warning: no HR flag found in file, using all gas particles")
        phrm = np.ones(len(pmass))
        
    hrmask = phrm > 0

    pdens = pdens[hrmask]
    pmass = pmass[hrmask]
    ptemp = ptemp[hrmask]

    if not hasattr(bins,'__len__'):
        bins = [bins, bins]

    if log_extent is None:
        extent = np.array([pdens.min(), pdens.max(), ptemp.min(), ptemp.max()])
        logext = np.log10(extent)
    else:
        logext = log_extent
    
    dbins = np.logspace(logext[0], logext[1], bins[0])
    Tbins = np.logspace(logext[2], logext[3], bins[1])

    X,Y = np.meshgrid(dbins, Tbins)

    HH = np.histogram2d(pdens,ptemp, bins=[dbins,Tbins], weights=pmass)

    result = HH[0] / pmass.sum()
    return result, X, Y


def load_abundances(f):
    if 'SGCHEM' in f['Config'].attrs.keys():
        chemistry_network = int(get_attribute(f,'Config','CHEMISTRYNETWORK'))

        chemical_abundances = load_dataset_from_parttype(f,0,'ChemicalAbundances', remove_h_factors=0)

        if chemistry_network == 1 or chemistry_network == 5 or chemistry_network == 10:
            return chemical_abundances
        
        if chemistry_network == 1:
            x_H2    = chemical_abundances[:,0]
            x_HII   = chemical_abundances[:,1]
            x_Dp    = chemical_abundances[:,2]
            x_HD    = chemical_abundances[:,3]
            x_Hep   = chemical_abundances[:,4]
            x_Hepp  = chemical_abundances[:,5]
        elif chemistry_network == 5 or chemistry_network == 10:
            x_H2    = chemical_abundances[:,0]
            x_HII   = chemical_abundances[:,1]
            x_CO    = chemical_abundances[:,2]
        else:
            raise ValueError("Unknown chemistry network: " + chemistry_network)
    else:
        raise ValueError("This run does not have chemistry")