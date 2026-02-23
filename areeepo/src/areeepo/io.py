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

import h5py
import numpy as np

# ============================================================= #
# I/O AND DATA HANDLING FUNCTIONS FOR HDF5 SNAPSHOT FILES
# ============================================================= #

# ----- Basic IO methods

def get_attribute(f,dsetpath,attrname):
    """Load arbitrary attribute from HDF5 file
    """
    try:
        attr = f[dsetpath].attrs[attrname]
        return attr
    except KeyError:
        raise KeyError("Attribute '" + attrname + "' in dataset or group '" + dsetpath + "' not found in HDF5 file.")

def get_h(f):
    """Load Hubble Parameter (little h) from HDF5 file
    """
    # Ensure we are at the root of the HDF5 file
    fr = f.file

    if 'HubbleParam' in fr['Header'].attrs.keys():
        h = fr['Header'].attrs['HubbleParam']
    elif 'Parameters' in fr.keys() and 'HubbleParam' in fr['Parameters'].attrs.keys():
        h = fr['Parameters'].attrs['HubbleParam']
    else:
        h = None
        raise KeyError("Hubble parameter not found in HDF5 file")
    return h

def load_dataset_from_hdf5_file(f,dsetpath,return_attrs = False, remove_h_factors = True, convert_to_cgs = False):
    """Load a HDF5 dataset and return it as a Numpy array

    Loads a dataset at the specified path from a HDF5 file, and returns it as
    a numpy array. Has options to convert the dataset to desired units.

    Parameters
    ----------
    f : h5py.File or h5py.Group
        HDF5 file
    dsetpath : string
        Path relative to the root of f where to find the dataset
    return_attrs : bool
        Also return the dataset's attributes as a dictionary. Default: False
    remove_h_factors : bool
        Return the dataset in "h-free" units. Default: True
    convert_to_cgs : bool
        Rerturn the dataset in CGS units. Default: False
    """
    try:
        data = np.array(f[dsetpath])
        attrs = dict(f[dsetpath].attrs)
    except KeyError:
        raise KeyError("Dataset " + dsetpath + " not found in HDF5 file " + f.filename)
    
    if remove_h_factors:
        if not 'h_scaling' in attrs:
            print("Warning: no h scaling found in dataset, using 1 by default")
            h_scaling = 1
        else:
            h_scaling = attrs['h_scaling']
        h_Hubble = get_h(f)
        data *= h_Hubble**(h_scaling)

    if convert_to_cgs:
        conv = attrs['to_cgs']
        data *= conv

    if return_attrs:
        return data, attrs
    else:
        return data

def load_header(f):
    """Load attributes of Header as a python dictionary
    """
    return dict(f['Header'].attrs)

def load_units(f):
    """Load AREPO units from a snapshot as a dictionary
    """
    if 'UnitMass_in_g' in f['Header'].attrs:
        ugkey = 'Header'
    else:
        ugkey = 'Parameters'
    UnitMass = f[ugkey].attrs['UnitMass_in_g']
    UnitLength = f[ugkey].attrs['UnitLength_in_cm']
    UnitVelocity = f[ugkey].attrs['UnitVelocity_in_cm_per_s']
    UnitTime = UnitLength / UnitVelocity
    UnitDensity = UnitMass / UnitLength**3
    UnitEnergy = UnitMass * UnitVelocity**2
    units = {
        'UnitMass' : UnitMass,
        'UnitLength' : UnitLength,
        'UnitVelocity' : UnitVelocity,
        'UnitTime' : UnitTime,
        'UnitDensity' : UnitDensity,
        'UnitEnergy' : UnitEnergy
    }
    return units

# ----- Wrappers for dataset-loading

def load_dataset_from_parttype(f,PartType,dsetname,return_attrs = False, remove_h_factors = True, convert_to_cgs = False):
    """Load dataset (or a list of datasets) for a given PartType
    """
    single_output = False
    if not isinstance(dsetname,list):
        dsetname = [dsetname]
        single_output = True
    res = []
    for dsn in dsetname:
        dsetpath = f"PartType{PartType:n}/" + dsn
        res.append( load_dataset_from_hdf5_file(f,dsetpath,return_attrs,remove_h_factors,convert_to_cgs) )

    if single_output: res = res[0]
    return res


def load_gas(f,dsetname,return_attrs = False, remove_h_factors = True, convert_to_cgs = False):
    """Load dataset (or a list of datasets) for PartType0 (gas)
    """
    return load_dataset_from_parttype(f,0,dsetname,return_attrs,remove_h_factors,convert_to_cgs)

def load_dm(f,dsetname,return_attrs = False, remove_h_factors = True, convert_to_cgs = False):
    """Load dataset (or a list of datasets) for PartType1 (dm)
    """
    return load_dataset_from_parttype(f,1,dsetname,return_attrs,remove_h_factors,convert_to_cgs)

def load_stars(f,dsetname,return_attrs = False, remove_h_factors = True, convert_to_cgs = False):
    """Load dataset (or a list of datasets) for PartType4 (stars)
    """
    return load_dataset_from_parttype(f,4,dsetname,return_attrs,remove_h_factors,convert_to_cgs)

# ----- Quick getters for other quantities & attributes

def get_time(f):
    """Load "time" from HDF5 file
    
    In non-cosmological simulations, this is the physical time in internal units.
    In cosmological simulations, it is the dimensionless scale-factor.
    """
    # Ensure we are at the root of the HDF5 file
    fr = f.file
    return fr['Header'].attrs['Time']

def get_redshift(f):
    a = get_time(f)
    return 1. / a - 1.0

def get_boxsize(f, remove_h = True):
    # Ensure we are at the root of the HDF5 file
    fr = f.file
    L = get_attribute(fr,'Header','BoxSize')
    h_scaling = 1.0
    if remove_h and is_cosmological(fr):
        h_scaling = 1. / get_h(fr)

    return L * h_scaling

def get_masstable(f, remove_h = True):
    fr = f.file
    mt = get_attribute(fr, 'Header', 'MassTable')
    h_scaling = 1.0
    if remove_h and is_cosmological(fr):
        h_scaling = 1. / get_h(fr)
    return mt * h_scaling

def get_numpart(f):
    fr = f.file
    return np.array(fr['Header'].attrs['NumPart_Total'])

def is_cosmological(f):
    """Check if snapshot is from a cosmological (comoving) run
    """
    fr = f.file
    return bool(get_attribute(fr,'Parameters','ComovingIntegrationOn'))

def is_zoom(f):
    """Check if snapshot is from a zoom-in simulation

    !! This assumes that we're using the PM_ZOOM_OPTIMIZED flag !!
    """
    fr = f.file
    if 'Config' in fr['Config'].attrs.keys():
        return True
    else:
        return False
    
# ============================================================= #
# PRINTING FUNCTIONS
# ============================================================= #

def print_hdf5_level(f,level,depth,attrs):       
    for i in range(level): print("  ",end="")
    print("└─ ",end="")

    print(f.name.split('/')[-1])
    if attrs:
        na = len(f.attrs)
        for i in range(level): print("  ",end="")
        print(f"{na:n} attribute(s):")
        for i in range(level): print("  ",end="")
        print(f.attrs.keys())
    if hasattr(f,'keys'):
        if level <= depth:
            for k in f.keys():
                print_hdf5_level(f[k],level+1,depth,attrs)

def print_hdf5_file(f,root='/',depth=1,attrs=False):
    print_hdf5_level(f[root],0,depth,attrs)