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

from scipy.spatial import KDTree
import numpy as np
from .utils import *

# ============================================================= #
# INTERPOLATION & PROJECTION METHODS, FOR VISUALISATIONS
# ============================================================= #

def grid_with_NN(pos, vals, bins, extent = None, **kwargs):
    """Interpolate discrete scalar field onto 3D cartesian grid using nearest neighbours

    Creates a 3D cartesian grid of square voxels, with size specified by extent and
    resolution specified by bins, and fills the grid using nearest-neighbour interpolation.

    `vals` can contain a list of different physical quantities to interpolate
    (e.g. density, temperature,...), which is significantly faster than calling
    the function multiple times on individual quantities, as the neighbour search
    has to be performed only once.

    Parameters
    ----------
    pos : array
        Positions of the base points
    vals : array or list of arrays
        Field value at the base points. If a list of arrays, the function grids multiple
        scalar fields.
    bins : array of shape (3,)
        Number of bins in each dimension
    extent : array of shape (3,2)
        Boundaries of the field to consider, in each dimension, in units of ``pos``
    tree : KDTree (optional)
        Instance of KDtree for the point set. If none given, is initialized by the function

    Returns
    -------
    outvals : array or list of arrays
        If ``vals`` is a single array, value of this field at the grid points. If it is a
        list of arrays, ``outvals`` is a list of arrays holding the interpolated fields.
    binsizes : array of shape (3,)
        Sizes of the 3 bins, i.e. [dx,dy,dz]
    """
    tree = kwargs.get('tree', None)

    if extent is None:
        extent = get_default_extent(pos)

    # Scalar bins
    if not hasattr(bins,'__len__'):
        bins = [bins, bins, bins]
    
    # Create cartesian grid (here these are bin centers)
    X, dx = np.linspace(extent[0,0],extent[0,1],bins[0],retstep=True, endpoint=False)
    Y, dy = np.linspace(extent[1,0],extent[1,1],bins[1],retstep=True, endpoint=False)
    Z, dz = np.linspace(extent[2,0],extent[2,1],bins[2],retstep=True, endpoint=False)
    
    # Shift to centers
    X += dx/2.0
    Y += dy/2.0
    Z += dz/2.0

    XX, YY, ZZ = np.meshgrid(X,Y,Z, indexing='ij')
    gpoints = np.stack((XX.flatten(),YY.flatten(),ZZ.flatten()), axis=1)
    
    # Create tree if not provided
    if tree is None:
        tree = KDTree(pos)
    
    # Find index of nearest neighbour to each grid point (voxel)
    dist, i = tree.query(gpoints)
    
    # Here we check if we're interpolating only one field or a list of fields
    single_output = False
    if not isinstance(vals,list):
        vals = [vals]
        single_output = True   

    # Loop over fields
    outvals = []
    for val in vals:
        outvals.append(val[i].reshape(bins, order='C'))
    
    # Return result
    binsizes = np.array([dx,dy,dz])
    if single_output:
        return outvals[0], binsizes
    else:
        return outvals, binsizes

def grid_with_histogram(pos, vals, bins, extent = None, **kwargs):
    """Bins a discrete scalar field in a 3D cartesian grid

    Creates a 3D cartesian grid of square voxels, with size specified by extent and
    resolution specified by bins, and fills the grid using a simple histogram (without
    any interpolation).

    Parameters
    ----------
    pos : array
        Positions of the base points
    vals : array or list of arrays
        Field value at the base points. If a list of arrays, the function grids multiple
        scalar fields.
    bins : array of shape (3,)
        Number of bins in each dimension
    extent : array of shape (3,2)
        Boundaries of the field to consider, in each dimension, in units of ``pos``
    density : bool (default: True)
        If true, divide the value in each bin by the volume of the bin. For example, when
        projecting the mass field, density=True will give the (binned) physical
        mass density
    
    Returns
    -------
    outvals : array or list of arrays
        If ``vals`` is a single array, value of this field at the grid points. If it is a
        list of arrays, ``outvals`` is a list of arrays holding the interpolated fields.
    binsizes : array of shape (3,)
        Sizes of the 3 bins, i.e. [dx,dy,dz]
    """
    density = kwargs.get('density', True)

    if extent is None:
        extent = get_default_extent(pos)

    # Scalar bins
    if not hasattr(bins,'__len__'):
        bins = [bins, bins, bins]

    # Create cartesian grid (these are the bin edges passed to histogram, so add +1)
    X, dx = np.linspace(extent[0,0],extent[0,1],bins[0] + 1,retstep=True)
    Y, dy = np.linspace(extent[1,0],extent[1,1],bins[1] + 1,retstep=True)
    Z, dz = np.linspace(extent[2,0],extent[2,1],bins[2] + 1,retstep=True)
    H, edges = np.histogramdd(pos, bins = (X,Y,Z), range = extent, weights = vals)

    if density:
        dV = dx * dy * dz
        H /= dV
    binsizes = np.array([dx,dy,dz])

    return H, binsizes

def project(pos,vals,axis,bins, gridding_function, extent=None, **kwargs):
    """Project a scalar field along an axis using a specified gridding function
    """
    if extent is None:
        extent = get_default_extent(pos)
        
    grid_vals, binsizes = gridding_function(pos,vals,bins,extent, **kwargs)
        
    single_output = False
    if not isinstance(grid_vals,list):
        grid_vals = [grid_vals]
        single_output = True

    out = []
    for v in grid_vals:
        out.append(_project_grid(v, binsizes, axis))

    extent_2D = np.delete(extent,axis,0).flatten()

    if single_output:
        return out[0], binsizes, extent_2D
    else:
        return out, binsizes, extent_2D

def project_with_NN(pos,vals,axis,bins,extent=None,tree=None):
    """Project a scalar field along an axis using nearest-neighbour interpolation

    Creates a 3D cartesian grid of square voxels, with size specified by extent and
    resolution specified by bins, fills the grid using nearest-neighbour interpolation
    and projects it along a specified axis.

    `vals` can contain a list of different physical quantities to interpolate
    (e.g. density, temperature,...), which is significantly faster than calling
    the function multiple times on individual quantities, as the neighbour search
    has to be performed only once.

    Parameters
    ----------
    pos : array
        Positions of the base points
    vals : array or list of arrays
        Field value at the base points. If a list of arrays, the function grids multiple
        scalar fields.
    axis: int
        Axis to project the grid along
    bins : array of shape (3,)
        Number of bins in each dimension
    extent : array of shape (3,2)
        Boundaries of the field to consider, in each dimension, in units of ``pos``
    tree : KDTree (optional)
        Instance of KDtree for the point set. If none given, is initialized by the function

    Returns
    -------
    out : array or list of arrays
        If ``vals`` is a single array, projected value of this field. If it is a
        list of arrays, ``outvals`` is a list of arrays holding the projected fields.
    binsizes : array of shape (3,)
        Sizes of the 3 bins, i.e. [dx,dy,dz]
    extent_2D : 1D array
        Flat array giving the edges of the projected field in the order required by `plt.imshow`
    """
    kwargs = {'tree' : tree}
    fgrid = grid_with_NN
    return project(pos, vals, axis, bins, fgrid, extent, **kwargs)

def project_with_histogram(pos,vals,axis,bins,extent=None,density=True):
    """Project a scalar field along an axis using a simple histogram

    Creates a 3D cartesian grid of square voxels, with size specified by extent and
    resolution specified by bins, fills the grid using a histogram
    and projects it along a specified axis.

    Parameters
    ----------
    pos : array
        Positions of the base points
    vals : array or list of arrays
        Field value at the base points. If a list of arrays, the function grids multiple
        scalar fields.
    axis: int
        Axis to project the grid along
    bins : array of shape (3,)
        Number of bins in each dimension
    extent : array of shape (3,2)
        Boundaries of the field to consider, in each dimension, in units of ``pos``
    density : bool (default: True)
        If true, divide the value in each bin by the volume of the bin. For example, when
        projecting the mass field, density=True will give the (binned) physical
        mass density

    Returns
    -------
    out : array or list of arrays
        If ``vals`` is a single array, projected value of this field. If it is a
        list of arrays, ``outvals`` is a list of arrays holding the projected fields.
    binsizes : array of shape (3,)
        Sizes of the 3 bins, i.e. [dx,dy,dz]
    extent_2D : 1D array
        Flat array giving the edges of the projected field in the order required by `plt.imshow`
    """
    kwargs = {'density' : density}
    fgrid = grid_with_histogram
    return project(pos, vals, axis, bins, fgrid, extent, **kwargs)


def _project_grid(grid_vals,binsizes,axis):
    return grid_vals.sum(axis=axis) * binsizes[axis]