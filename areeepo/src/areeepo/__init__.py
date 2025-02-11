import h5py
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

mP_cgs = 1.6726231e-24
kB_cgs = 1.380658e-16
kpc_cgs = 3.0857e21

def load_h_from_hdf5_file(f):
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

def get_time(f):
    # Ensure we are at the root of the HDF5 file
    fr = f.file
    return fr['Header'].attrs['Time']
    
def load_dataset_from_hdf5_file(f,dsetpath,return_attrs = False, remove_h_factors = True, convert_to_cgs = False):
    try:
        data = np.array(f[dsetpath])
        attrs = dict(f[dsetpath].attrs)
    except KeyError:
        raise KeyError("Dataset " + dsetpath + " not found in HDF5 file.")
    
    if remove_h_factors:
        if not 'h_scaling' in attrs:
            print("Warning: no h scaling found in dataset, using 1 by default")
            h_scaling = 1
        else:
            h_scaling = attrs['h_scaling']
        h_Hubble = load_h_from_hdf5_file(f)
        data *= h_Hubble**(h_scaling)

    if convert_to_cgs:
        conv = attrs['to_cgs']
        data *= conv

    if return_attrs:
        return data, attrs
    else:
        return data

def load_attribute_from_hdf5_file(f,dsetpath,attrname):
    try:
        attr = f[dsetpath].attrs[attrname]
        return attr
    except KeyError:
        raise KeyError("Attribute '" + attrname + "' in dataset or group '" + dsetpath + "' not found in HDF5 file.")

def load_dataset_from_parttype(f,PartType,dsetname,return_attrs = False, remove_h_factors = True, convert_to_cgs = False):
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
    return load_dataset_from_parttype(f,0,dsetname,return_attrs,remove_h_factors,convert_to_cgs)

def load_header_from_hdf5_file(f):
    return dict(f['Header'].attrs)


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

def load_abundances(f):
    if 'SGCHEM' in f['Config'].attrs.keys():
        chemistry_network = int(load_attribute_from_hdf5_file(f,'Config','CHEMISTRYNETWORK'))

        chemical_abundances = load_dataset_from_parttype(f,0,'ChemicalAbundances')

        if chemistry_network == 1 or chemistry_network == 5:
            return chemical_abundances
        
        if chemistry_network == 1:
            x_H2    = chemical_abundances[:,0]
            x_HII   = chemical_abundances[:,1]
            x_Dp    = chemical_abundances[:,2]
            x_HD    = chemical_abundances[:,3]
            x_Hep   = chemical_abundances[:,4]
            x_Hepp  = chemical_abundances[:,5]
        elif chemistry_network == 5:
            x_H2    = chemical_abundances[:,0]
            x_HII   = chemical_abundances[:,1]
            x_CO    = chemical_abundances[:,2]
        else:
            raise ValueError("Unknown chemistry network: " + chemistry_network)
    else:
        raise ValueError("This run does not have chemistry")

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

def compute_mean_molecular_weight(f):
    if 'SGCHEM' in f['Config'].attrs.keys():
        chemistry_network = int(load_attribute_from_hdf5_file(f,'Config','CHEMISTRYNETWORK'))
        chemical_abundances = load_dataset_from_parttype(f,0,'ChemicalAbundances', remove_h_factors=False) # ChemicalAbundances has no h scaling
        # TODO: Add option for variable Z
        ZAtom = float(f['Parameters'].attrs['ZAtom']) # Global metallicity, relative to solar
        
        if chemistry_network == 1:
            x_H2    = chemical_abundances[:,0]
            x_HII   = chemical_abundances[:,1]
            x_Dp    = chemical_abundances[:,2]
            x_HD    = chemical_abundances[:,3]
            x_Hep   = chemical_abundances[:,4]
            x_Hepp  = chemical_abundances[:,5]
        elif chemistry_network == 5:
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
        if chemistry_network == 5:
            x_e = x_HII
            frac_sum = x_HI + x_HII + x_H2 + x_e + x_He + x_CO
        elif chemistry_network == 1:
            x_e = x_HII + x_Hep + 2*x_Hepp
            frac_sum = x_HI + x_HII + x_H2 + x_e + x_He + x_Dp + x_HD + x_Hep + x_Hepp
        
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

def load_units(f):
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

def compute_temperature(f, gamma = 5.0/3.0):
    mu = compute_mean_molecular_weight(f)
    u = load_dataset_from_parttype(f,0,'InternalEnergy')
    units = load_units(f)
    T = mP_cgs * mu * (gamma-1) * u * units['UnitEnergy']/units['UnitMass']/kB_cgs
    return T


def grid_with_NN(pos,vals,bins,extent=None,tree = None):
    """Interpolate discrete scalar field onto 3D cartesian grid using nearest neighbours

    Creates a 3D cartesian grid of square voxels, with size specified by extent and
    resolution specified by bins, and fills the grid using nearest-neighbour interpolation.

    Parameters
    ----------
    pos : array
        Positions of the base points
    vals : array or list of arrays
        Field value at the base points. If a list of arrays, interpolates multiple scalar
        fields (e.g. density, temperature,...)
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
    if extent is None:
        extent = np.empty((3,3))
        extent[0,0] = pos[:,0].min()
        extent[0,1] = pos[:,0].max()
        extent[1,0] = pos[:,1].min()
        extent[1,1] = pos[:,1].max() 
        extent[2,0] = pos[:,2].min()
        extent[2,1] = pos[:,2].max() 

    # Scalar bins
    if not hasattr(bins,'__len__'):
        bins = [bins, bins, bins]
    
    # Create cartesian grid
    X, dx = np.linspace(extent[0,0],extent[0,1],bins[0],retstep=True)
    Y, dy = np.linspace(extent[1,0],extent[1,1],bins[1],retstep=True)
    Z, dz = np.linspace(extent[2,0],extent[2,1],bins[2],retstep=True)
    XX, YY, ZZ = np.meshgrid(X,Y,Z)
    gpoints = np.transpose(np.stack((XX.flatten(),YY.flatten(),ZZ.flatten())))
    
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
        outvals.append(val[i].reshape(bins))
    
    # Return result
    binsizes = np.array([dx,dy,dz])
    if single_output:
        return outvals[0], binsizes
    else:
        return outvals, binsizes

def project_with_NN(pos,vals,axis,bins,extent=None,tree=None):
    """Project discrete scalar field onto 2D cartesian grid using nearest neighbours

    [Description]

    Parameters
    ----------
    pos : array
        Positions of the base points
    vals : array or list of arrays
        Field value at the base points. If a list of arrays, interpolates multiple scalar
        fields (e.g. density, temperature,...)
    axis : int
        Axis along which to project the field
    bins : array of shape (3,)
        Number of bins in each dimension
    extent : array of shape (3,2)
        Boundaries of the field to consider, in each dimension, in units of ``pos``.
        In particular, ``extent[axis,:]`` determines the depth (limits) of the projection
    tree : KDTree (optional)
        Instance of KDtree for the point set. If none given, is initialized by the function

    Returns
    -------
    proj : array or list of arrays
        If ``vals`` is a single array, projection of this field. If it is a list of arrays,
        ``proj`` is a list of arrays holding the projected fields.
    binsizes : array of shape (2,)
        Sizes of the 2 bins, i.e. [dx,dy]
    """
    if extent is None:
        extent = np.empty((3,3))
        extent[0,0] = pos[:,0].min()
        extent[0,1] = pos[:,0].max()
        extent[1,0] = pos[:,1].min()
        extent[1,1] = pos[:,1].max() 
        extent[2,0] = pos[:,2].min()
        extent[2,1] = pos[:,2].max() 
        
    grid_vals, binsizes = grid_with_NN(pos,vals,bins,extent,tree)

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

def _project_grid(grid_vals,binsizes,axis):
    return grid_vals.sum(axis=axis) * binsizes[axis]


def crop(pos, extent):
    mask_x = np.logical_and(pos[:,0] >= extent[0,0] , pos[:,0] < extent[0,1])
    mask_y = np.logical_and(pos[:,1] >= extent[1,0] , pos[:,1] < extent[1,1])
    mask_z = np.logical_and(pos[:,2] >= extent[2,0] , pos[:,2] < extent[2,1])
    full_mask = np.logical_and( np.logical_and(mask_x, mask_y) , mask_z)
    return pos[full_mask], full_mask

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

def make_image_gas(f, bins, quantities = 'density', extent = None, axis = 2, remove_h_factors = True):
    single_output = False
    if not isinstance(quantities,list):
        quantities = [quantities]
        single_output = True
    
    pos = load_dataset_from_parttype(f,0,'Coordinates',remove_h_factors=remove_h_factors) / 1000
    fields = []

    # We always project the density, as it is needed for other mass-weighted quantities
    dens = load_dataset_from_parttype(f,0,'Density',remove_h_factors=remove_h_factors)
    fields.append(dens)

    if 'temperature' in quantities:
        temp = compute_temperature(f)
        fields.append(temp * dens) # Mass-weighted temperature
    
    # Only one call to projection method (so we only need to evaluate NN once for all fields)
    proj, bs, e2d = project_with_NN(pos,fields,axis,bins,extent)
    
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

#X,Y = np.meshgrid(dbins, Tbins)
#plt.pcolormesh(dbins,Tbins,HH[0].T, norm='log', cmap='cmr.ocean_r')
#plt.xscale('log')
#plt.yscale('log')
#plt.colorbar()
# from .io import *
# from .visualise import *
# from .utils import *