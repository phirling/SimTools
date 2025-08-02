import h5py
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

mP_cgs = 1.6726231e-24
kB_cgs = 1.380658e-16
kpc_cgs = 3.0857e21
Myr_cgs = 31557600000000.0

# ============================================================= #
# I/O AND DATA HANDLING FUNCTIONS FOR HDF5 SNAPSHOT FILES
# ============================================================= #

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

def load_header(f):
    """Load attributes of Header as a python dictionary
    """
    return dict(f['Header'].attrs)

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

# ============================================================= #
# FUNCTIONS TO COMPUTE DERIVED QUANTITIES FROM SNAPSHOTS
# ============================================================= #

def compute_mean_molecular_weight(f):
    if 'SGCHEM' in f['Config'].attrs.keys():
        chemistry_network = int(get_attribute(f,'Config','CHEMISTRYNETWORK'))
        chemical_abundances = load_dataset_from_parttype(f,0,'ChemicalAbundances', remove_h_factors=False) # ChemicalAbundances has no h scaling
        # TODO: Add option for variable Z
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
    mu = compute_mean_molecular_weight(f)
    u = load_dataset_from_parttype(f,0,'InternalEnergy')
    units = load_units(f)
    T = mP_cgs * mu * (gamma-1) * u * units['UnitEnergy']/units['UnitMass']/kB_cgs
    return T

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

# ============================================================= #
# INTERPOLATION & PROJECTION METHODS, FOR VISUALISATIONS
# ============================================================= #

def grid_with_NN(pos, vals, bins, extent = None, **kwargs):
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
    tree = kwargs.get('tree', None)

    if extent is None:
        extent = _default_extent(pos)

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
    print(gpoints.shape, gpoints)
    print(XX.flatten()[0:20])
    print(gpoints[:20, 0])
    
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
    """
    if density = True, we divide by the volume of each voxel to obtain a density (e.g. for mass)
    """
    density = kwargs.get('density', True)

    if extent is None:
        extent = _default_extent(pos)

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

    if extent is None:
        extent = _default_extent(pos)
        
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
    """Wrapper for project using nearest-neighbour interpolation
    """
    kwargs = {'tree' : tree}
    fgrid = grid_with_NN
    return project(pos, vals, axis, bins, fgrid, extent, **kwargs)

def project_with_histogram(pos,vals,axis,bins,extent=None,density=True):
    """Wrapper for project using a bare histogram
    """
    kwargs = {'density' : density}
    fgrid = grid_with_histogram
    return project(pos, vals, axis, bins, fgrid, extent, **kwargs)


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

# ============================================================= #
# UTILITY FUNCTIONS
# ============================================================= #
def idx_in_extent(pos, extent):
    mask_x = np.logical_and(pos[:,0] >= extent[0,0] , pos[:,0] < extent[0,1])
    mask_y = np.logical_and(pos[:,1] >= extent[1,0] , pos[:,1] < extent[1,1])
    mask_z = np.logical_and(pos[:,2] >= extent[2,0] , pos[:,2] < extent[2,1])
    full_mask = np.logical_and( np.logical_and(mask_x, mask_y) , mask_z)
    return full_mask

def crop(pos, extent):
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

def _project_grid(grid_vals,binsizes,axis):
    return grid_vals.sum(axis=axis) * binsizes[axis]

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

def _default_extent(pos):
    extent = np.empty((3,2))
    extent[0,0] = pos[:,0].min()
    extent[0,1] = pos[:,0].max()
    extent[1,0] = pos[:,1].min()
    extent[1,1] = pos[:,1].max() 
    extent[2,0] = pos[:,2].min()
    extent[2,1] = pos[:,2].max()
    return extent

def _select_zoom_DM_types(dm_pos, dm_weights, which):
    """Wrapper to select one of the three DM types in a zoom-in and build output array
    """
    DMPOS = None
    W = None
    
    if which == 'all':
        which = ['hr','or','lr']
        
    if 'hr' in which:
        DMPOS = dm_pos[0]
        W = dm_weights[0]
    if 'or' in which:
        if DMPOS is None:
            DMPOS = dm_pos[1]
            W = dm_weights[1]
        else:
            DMPOS = np.vstack((DMPOS, dm_pos[1]))
            W = np.hstack((W, dm_weights[1]))
    if 'lr' in which:
        if DMPOS is None:
            DMPOS = dm_pos[2]
            W = dm_weights[2]
        else:
            DMPOS = np.vstack((DMPOS, dm_pos[2]))
            W = np.hstack((W, dm_weights[2]))
    if DMPOS is None:
        raise ValueError("Need to select one type of DM")
    
    return DMPOS, W

# ============================================================= #
# EXPERIMENTAL
# ============================================================= #

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

        chemical_abundances = load_dataset_from_parttype(f,0,'ChemicalAbundances')

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