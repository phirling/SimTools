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
from .io import *

# Hardcoded physical constants in CGS units
mP_cgs = 1.6726231e-24
kB_cgs = 1.380658e-16
kpc_cgs = 3.0857e21
Myr_cgs = 31557600000000.0

# ============================================================= #
# UTILITY FUNCTIONS
# ============================================================= #

def in_extent(pos, extent):
    """Checks if `pos` is inside `extent` (in a euclidian 3D sense)

    Handles different shapes for the pos array internally.
    """
    if len(pos.shape) < 2:
        tpos = np.reshape(pos, (1,3))
        scalar_output = True
    else:
        tpos = pos
        scalar_output = False

    mask_x = np.logical_and(tpos[:,0] >= extent[0,0] , tpos[:,0] < extent[0,1])
    mask_y = np.logical_and(tpos[:,1] >= extent[1,0] , tpos[:,1] < extent[1,1])
    mask_z = np.logical_and(tpos[:,2] >= extent[2,0] , tpos[:,2] < extent[2,1])
    full_mask = np.logical_and( np.logical_and(mask_x, mask_y) , mask_z)
    if scalar_output:
        return full_mask[0]
    else:
        return full_mask

def crop(pos, extent):
    """Return a copy of `pos` that only contains entries inside `extent`
    """
    full_mask = idx_in_extent(pos, extent)
    return pos[full_mask]

def crop_dimension(pos, extent, dim = 2):
    mask_dim = np.logical_and(pos[:,dim] >= extent[0], pos[:,dim] < extent[1])
    return mask_dim

def mode2dim(mode):
    if mode == 'xy':
        dim0 = 0
        dim1 = 1
        dim2 = 2
    elif mode == 'xz':
        dim0 = 0
        dim1 = 2
        dim2 = 1
    elif mode == 'yz':
        dim0 = 1
        dim1 = 2
        dim2 = 0
    else:
        raise ValueError("Invalid Mode chosen:" + mode)
    return dim0, dim1, dim2

def get_default_extent(pos):
    """Compute minimum encompassing extent of a 3D point distribution
    """
    extent = np.empty((3,2))
    extent[0,0] = pos[:,0].min()
    extent[0,1] = pos[:,0].max()
    extent[1,0] = pos[:,1].min()
    extent[1,1] = pos[:,1].max() 
    extent[2,0] = pos[:,2].min()
    extent[2,1] = pos[:,2].max()
    return extent

def get_zoom_gas_mask(f):
    """Return mask of high-resolution gas cells for zoom-in simulations
    """
    try:
        phrm = load_gas(f,'HighResGasMass', remove_h_factors=0)
        hrmask = phrm > 0
    except KeyError:
        # Print a warning when we're loading gas in a zoom simulation
        print(f"Warning: no HR flag found in file {f.filename}, using all gas particles")
        NumPartGas = get_numpart(f)[0]
        hrmask = np.ones(NumPartGas, dtype='bool')
    return hrmask

# ----- Deprecated functions

def idx_in_extent(pos, extent):
    """Legacy name
    """
    return in_extent(pos, extent)