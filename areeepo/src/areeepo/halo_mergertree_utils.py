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
import h5py
from .io import *
from .utils import *

# ============================================================= #
# FOF, SUBHALO AND MERGER TREE UTILITIES
# ============================================================= #
def get_subhalo_ascendence(groupcat_basename, snap_range, i_subhalo, flex = False, verbose = False):
    """Find indices of main progenitors of subhalo back in time

    Parameters
    ----------
    groupcat_basename : str
        Basename of the subhalo progenitor files, relative to working directory. They are
        internally appended by e.g. '_001.hdf5'. Example: './output/subhalo_prog_'
    snap_range : iterable
        Range of snapshots over which the history is to be extracted. The last element
        is assumed to be where the target subhalo is selected.
    i_subhalo : int
        Index of target subhalo in the catalogue at snap_range[-1]. The main subhalo is
        i_subhalo = 0.
    flex : bool
        Flexible ascendence length. If true, the ascendence list can be stopped earlier
        than asked by snap_range without throwing an error. 'snap_range' is then used as
        an upper bound only.
        
    Returns
    -------
    subhalo_indices : list
        List of indices into subhalo table of the main progenitor branch. Order is identical to snap_range
    """
    subhalo_indices = []

    jsub = i_subhalo

    inv_iterator = reversed(snap_range)
    last_snap = snap_range[0]
    
    for i in inv_iterator:
        subhalo_indices.append(jsub)
        if i != last_snap:
            file_current = groupcat_basename + f"_{i:03n}.hdf5"
            file_next = groupcat_basename + f"_{i-1:03n}.hdf5"
            with h5py.File(file_current) as f:
                prog_nr = f['Subhalo']['FirstProgSubhaloNr'][jsub]
                if prog_nr == -1:
                    if flex:
                        break
                    else:
                        raise ValueError(f"End of branch reached. The subhalo has no progenitor in {file_next}. Please change snap_range to e.g. ({i+1:n},{snap_range[-1]:n})")
            
            with h5py.File(file_next) as f:
                snumbers = f['Subhalo']['SubhaloNr'][:]
                jsub = np.where(snumbers == prog_nr)[0][0]

    if flex and verbose and i != snap_range[0]:
        print(f"Flexible mode: stopped ascendence at {i:n} for subhalo {i_subhalo:n}")
        
    subhalo_indices = list(reversed(subhalo_indices))
    return subhalo_indices

def load_subhalo_data(fgc, dsetname):
    return np.array(fgc['Subhalo'][dsetname])

def get_idx_bound_to_subhalo(fgc, i_sub, PartType):
    """
    Helper function to get the indices of particles bound to a given subhalo, for a given particle type

    Parameters
    ----------
    fof_f : h5py.File
        Group catalogue file
    idx_subhalo : int
        Index of the SUBFIND subhalo
    PartType : int
        Particle type (0-5)

    Returns
    -------
    idx : np.array
        Array of indices into the particle dataset in the snapshot file
    """
    offsets = fgc['Subhalo']['SubhaloOffsetType'][i_sub]
    lengths = fgc['Subhalo']['SubhaloLenType'][i_sub]

    idx = np.arange(offsets[PartType], offsets[PartType] + lengths[PartType])
    return idx

def get_idx_in_radius_subhalo(fsnap, fgc, i_sub, radius, PartType, gas_only_refined = True):
    """

    radius is given in h-free, comoving units
    """
    h = get_h(fsnap)
    spos = load_subhalo_data(fgc, 'SubhaloPos') / h

    centered_pos = load_dataset_from_parttype(fsnap, PartType, 'Coordinates', remove_h_factors=True) - spos[i_sub]

    if PartType == 0 and gas_only_refined:
        hrmask = get_zoom_gas_mask(fsnap)
        centered_pos = centered_pos[hrmask]

    idx = np.where(np.linalg.norm(centered_pos, axis=1) <= radius)
    return idx[0]