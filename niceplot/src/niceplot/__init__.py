from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Paths
module_root = Path(__file__).parent
style_root = module_root / "styles"

# ============================================================= #
# METHODS TO SET MATPLOTLIB STYLESHEET & OTHER CUSTOM SETTINGS
# ============================================================= #

def enable_latex(plt):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        #"figure.figsize" : (5.0, 4.3)
    })

def disable_latex(plt):
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif"
    })

def set_default_style(plt):
    plt.style.use('default')

def set_bismuth_style(plt):
    spath = style_root / "bismuth.mplstyle"
    plt.style.use(spath)

def set_style(plt, stylename = 'bismuth', latex = False):
    match stylename:
        case 'default':
            set_default_style(plt)
        case 'bismuth':
            set_bismuth_style(plt)
        case _:
            raise ValueError("Unknown style name: " + str(stylename))

    if latex:
        enable_latex(plt)
    else:
        disable_latex(plt)

def reset_style(plt):
    set_default_style(plt)
    disable_latex(plt)

# ============================================================= #
# PLOT DECORATION METHODS
# ============================================================= #

def add_redshift_axis(ax, zlist, label = 'Redshift'):
    """Add secondary top redshift axis to a scale-factor-dependent plot
    """
    a2z = lambda a : 1./a-1.
    z2a = lambda z : 1./(z+1)
    secx = ax.secondary_xaxis('top', functions=(a2z, z2a), xlabel=label)
    secx.set_xticks(zlist)
    ax.tick_params('x', reset=1, top=0)
    #secx.set_xticklabels([f'{x}' for x in secx.get_xticks()])
    return secx
    
def add_nice_colorbar(im, location = 'right', label = None,
                     pad = 0.01, thickness = 0.03):
    """Add a nicely-spaced colorbar to an existing Axes/AxesImage pair
    """
    ax = im.axes
    fig = ax.figure
    
    if location == 'right':
        cax = fig.add_axes([ax.get_position().x1+pad,ax.get_position().y0, thickness,ax.get_position().height])
    else:
        cax = fig.add_axes([ax.get_position().x0,ax.get_position().y1+pad,ax.get_position().width, thickness])
    cax.tick_params(axis='both',direction='out',reset=True)
    
    cbar = plt.colorbar(im, cax = cax, location=location, label=label)
    return cbar

def add_info_text(ax, text, loc = 'ul', color ='white', fontsize = 8):
    """Add info text in corner of axes

    Parameters
    ----------
    ax : Axes
        Axes upon which to draw
    text : string
        The text to write
    loc : string
        Corner in which to draw. Can be ('ll','lr','ul' or 'ur'). (default: upper left)
    color : string
        Color of the text (default: white)
    fontsize : float
        Fontsize (default: 8)
    """
    # Lower Left
    if loc == 'll':
        x = 0.03
        y = 0.03
        ha = 'left'
        va = 'bottom'
    # Lower Right
    elif loc == 'lr':
        x = 0.97
        y = 0.03
        ha = 'right'
        va = 'bottom'
    # Upper Left
    elif loc == 'ul':
        x = 0.03
        y = 0.97
        ha = 'left'
        va = 'top'
    # Upper Right
    elif loc == 'ur':
        x = 0.97
        y = 0.97
        ha = 'right'
        va = 'top'
    else:
        raise ValueError("Unknown location: ", loc)
        
    return ax.text(x,y,text,ha=ha,va=va,transform=ax.transAxes, color=color,fontsize=fontsize)

def add_object_markers(ax, pos, rad, label = "obj", label_numbering = True):
    ax.scatter(pos[:,0], pos[:,1])
    fr = 0.8
    for k in range(len(rad)):
        xy = (pos[k, 0],pos[k, 1])
        circ = Circle(xy , rad[k], fill = False)
        ax.add_patch(circ)
        lbl = label
        if label_numbering:
            lbl += f"{k:n}"
        ax.text(pos[k, 0] + fr*rad[k], pos[k, 1] + fr*rad[k], lbl , fontsize=10, color='white', fontfamily='monospace')

# ============================================================= #
# FULL FIGURE METHODS & WRAPPERS
# ============================================================= #

def imshow(data, xlabel = None, ylabel = None, cmap = None, label = None,
           extent = None, norm = None, interp = None,
           vmin = None, vmax = None,
           cbar_loc = 'right', s = 1.0, infotext = None, colorbar = True):
    """Wrapper for the Matplotlib imshow function, for better looking image plots
    """
    # The s argument is used to globally adjust the figure size
    # Figure Size
    H = 2.8 * s
    L = 3.5 * s
    
    # Colorbar Placement & Size
    pad = 0.02
    th = 0.05
    
    fig, ax = plt.subplots(1,1, figsize=(L,H))
    ax.set_aspect('equal', 'box')
    
    # Add x/y labels if necessary
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Now we create the image
    im = ax.imshow(data, cmap = cmap, norm = norm, vmin = vmin, vmax = vmax,
                   interpolation=interp, origin='lower', extent=extent)
    if colorbar:
        cbar = add_nice_colorbar(im, cbar_loc, pad=pad, thickness=th, label=label)
    else:
        cbar = None
        
    if infotext is not None:
        add_info_text(ax, infotext, loc='ul', color='white')
        
    return fig, ax, im, cbar

def pcolormesh(X,Y,data, xlabel = None, ylabel = None, cmap = None, label = None,
               xscale = 'log', yscale = 'log', norm = None, vmin = None, vmax = None,
               cbar_loc = 'right', s = 1.0):
    """Wrapper for the Matplotlib pcolormeshfunction, for better looking histograms
    """
    # The s argument is used to globally adjust the figure size
    # Figure Size
    H = 3.5 * s
    L = 3.5 * s
    
    # Colorbar Placement & Size
    pad = 0.02
    th = 0.05
    
    fig, ax = plt.subplots(1,1, figsize=(L,H))
    
    # Add x/y labels if necessary
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Scale
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    # Now we create the image
    im = ax.pcolormesh(X,Y,data, cmap = cmap, norm = norm, vmin = vmin, vmax = vmax)
    cbar = add_nice_colorbar(im, cbar_loc, pad=pad, thickness=th, label=label)

    return fig, ax, im, cbar
    
def imshow_grid(nrows, ncols, data, **kwargs):
    """Create a nice grid of images from data, with colorbars & labels

    Arguments
    ---------
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    data : list of data arrays

    Returns
    -------
    fig : matplotlib.figure
        Figure
    ax : matplotlib.axes
        Axes
    images : 2D list of matplotlib.AxesImage
        Grid of images created
    images : 2D list of matplotlib.colorbar
        Grid of color bars
    """
    return _make_grid_figure(nrows, ncols, data, 'imshow', **kwargs)

def pcolormesh_grid(nrows, ncols, X,Y, data, **kwargs):
    """
    """
    return _make_grid_figure(nrows, ncols, data, 'pcolormesh', X=X, Y=Y, **kwargs)


# ============================================================= #
# INTERNAL UTILITY METHODS
# ============================================================= #

def _setup_plotgrid(nrows, ncols, xlabel = None, ylabel = None, sharex = False, sharey = False, s = 1.0):
    """Initialize grid of figures for imshow or pcolormesh, with nice spacing
    """
    # ------------- Base Geometric Values. Only change this in worst case!
    # The s argument is used to globally adjust the figure size
    # Figure Vertical
    H1 = 3.5 * s
    bottom1 = 0.11
    top1 = 0.88
    hspace1 = 0.2
    fh = 1.05
    
    # Figure Horizontal
    L1 = 2.8 * s
    left1 = 0.125
    right1 = 0.9
    wspace1 = 0.2
    fl = 1.15
    
    # Colorbar Placement & Size
    pad1 = 0.02
    th1 = 0.05

    # ------------- Adjustments to grid shape
    # Main geometric adjustment factor
    hfact = 1 + (nrows-1) * fh
    lfact = 1 + (ncols-1) * fl

    # Figure Size
    H = H1 * hfact
    L = L1 * lfact

    # Figure Margins
    bottom = bottom1 / hfact
    top = 1 - (1-top1) / hfact
    left = left1 / hfact
    right = 1 - (1-right1) / hfact

    # Subplot padding
    if sharex:
        hspace = hspace1 / 1.5
    elif xlabel is None:
        hspace = hspace1
    else:
        hspace = hspace1 * 2.1
    
    if sharey:
        wspace = wspace1 / 1.5
    elif ylabel is None:
        wspace = wspace1
    else:
        wspace = wspace1 * 1.5

    # Colorbar padding
    pad = pad1 / hfact
    th = th1 / hfact

    # ------------- Now we create and setup the figure
    fig, ax = plt.subplots(nrows,ncols, figsize=(L,H), squeeze=False, sharex=sharex, sharey=sharey)
    
    # Adjust subplot margins
    fig.subplots_adjust(bottom=bottom,top=top,hspace=hspace,left=left, right=right, wspace=wspace)

    def select_prop(prop,i,j):
        if prop is None:
            return None
        elif isinstance(prop,list):
            return prop[i][j]
        else:
            return prop
        
    # Add x/y labels if necessary
    for i in range(nrows):
        for j in range(ncols):
            if not (sharex and i != nrows - 1):
                ax[i,j].set_xlabel(select_prop(xlabel,i,j))
            if not (sharey and j != 0):
                ax[i,j].set_ylabel(select_prop(ylabel,i,j))
        
    return fig, ax, pad, th

def _make_grid_figure(nrows, ncols, data, funcname, X=None, Y=None, **kwargs):
    """Create a nice grid of images from data, with colorbars & labels
    """
    
    # ------------- Check Data size
    np_data = False
    # By default, we assume that data & co are 2D lists of numpy arrays.
    if not isinstance(data,list):
        # Its also possible to pass a 4D numpy array
        if len(data.shape) == 4:
            np_data = True
        # Or a single array, for nrows = ncols = 1. Need to adapt in this case
        elif len(data.shape) == 2:
            data = [[data]]
            cmaps = [[cmaps]]
            labels = [[labels]]
        else:
            raise ValueError("Unknown data grid type")         
        
    assert len(data) == nrows and len(data[0]) == ncols

    # ------------- Unpack arguments
    s = kwargs.get('s', 1.0)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)

    normalize = kwargs.get('normalize', False)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)

    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')

    colorbar            = kwargs.get('colorbar', True)
    image_label_list    = kwargs.get('label', None)
    cmap_list           = kwargs.get('cmap', None)
    norm_list           = kwargs.get('norm', None)
    interp_list         = kwargs.get('interp', None)

    grid_extent         = kwargs.get('grid_extent', False)
    extent              = kwargs.get('extent', None)

    infotext            = kwargs.get('infotext', None)
    infotext_color      = kwargs.get('infotext_color', 'white')
    infotext_fontsize   = kwargs.get('infotext_fontsize', 8)
    infotext_location   = kwargs.get('infotext_location', 'ul')

    # ------------- Normalize Data if desired
    vmin_glob = None
    vmax_glob = None
    if normalize:
        if np_data:
            vmin_glob = data.min()
            vmax_glob = data.max()
        # If data is 2D list of 2D numpy arrays, we need to loop through
        else:
            vmin_glob = np.min(data[0][0])
            vmax_glob = np.max(data[0][0])
            for i in range(nrows):
                for j in range(ncols):
                    vmin_glob = min(vmin_glob,np.min(data[i][j]))
                    vmax_glob = max(vmax_glob,np.max(data[i][j]))

    if vmin is None: vmin = vmin_glob
    if vmax is None: vmax = vmax_glob
        
    # ------------- Create and setup figure
    fig, ax, pad, th = _setup_plotgrid(nrows,ncols,xlabel,ylabel,sharex,sharey,s)

    def select_prop(prop,i,j):
        if prop is None:
            return None
        #elif hasattr(prop,'__len__'):
        elif isinstance(prop,list):
            return prop[i][j]
        elif isinstance(prop, np.ndarray):
            return prop[i,j]
        else:
            return prop
    
    # Now we create the images
    images = []
    colorbars = []
    for i in range(nrows):
        imline = []
        cbline = []
        for j in range(ncols):
            if data[i][j] is None:
                imline = None
                cbar = None
                ax[i][j].set_axis_off()
            else:
                cmp = select_prop(cmap_list,i,j)
                lbl = select_prop(image_label_list,i,j)
                nrm = select_prop(norm_list,i,j)
                itp = select_prop(interp_list,i,j)
                vmn = select_prop(vmin,i,j)
                vmx = select_prop(vmax,i,j)
                if grid_extent:
                    ext = extent[i][j]
                else:
                    ext = extent

                #viz_function = getattr(ax[i][j], funcname)
                #im = viz_function(data[i][j], cmap = cmp, norm = nrm, vmin = vmn, vmax = vmx,
                #                     extent = ext, interpolation = itp, origin='lower')

                if funcname == 'imshow':
                    im = ax[i][j].imshow(data[i][j], cmap = cmp, norm = nrm, vmin = vmn, vmax = vmx, extent = ext, interpolation = itp, origin='lower')
                    ax[i][j].set_aspect('equal', 'box')
                elif funcname == 'pcolormesh' and X is not None and Y is not None:
                    im = ax[i][j].pcolormesh(X[i][j],Y[i][j],data[i][j], cmap = cmp, norm = nrm, vmin = vmn, vmax = vmx)
                    ax[i][j].set_box_aspect(1)
                else:
                    raise ValueError("Unknown function name: ", funcname)
                
                # Scale
                ax[i][j].set_xscale(xscale)
                ax[i][j].set_yscale(yscale)

                if colorbar:
                    cbar = add_nice_colorbar(im, 'top', pad=pad, thickness=th, label=lbl)
                else:
                    cbar = None

                if infotext is not None:
                    add_info_text(ax[i][j], infotext, loc=infotext_location, color=infotext_color, fontsize=infotext_fontsize)
                    
            imline.append(im)
            cbline.append(cbar)
            
        images.append(imline)
        colorbars.append(cbline)

    return fig, ax, images, colorbars

# ============================================================= #
# EXPERIMENTAL
# ============================================================= #

"""
def pcolormesh_grid(nrows, ncols, X,Y, data,
                    xlabel = None, ylabel = None, sharex = False, sharey = False,
                    xscale = 'log', yscale = 'log',
                    cmap = None, label = None, norm = None,
                    vmin = None, vmax = None, normalize = False, s = 1.0):

    # ------------- Check Data size
    np_data = False
    # By default, we assume that data & co are 2D lists of numpy arrays.
    if not isinstance(data,list):
        # Its also possible to pass a 4D numpy array
        if len(data.shape) == 4:
            np_data = True
        # Or a single array, for nrows = ncols = 1. Need to adapt in this case
        elif len(data.shape) == 2:
            data = [[data]]
            cmaps = [[cmaps]]
            labels = [[labels]]
        else:
            raise ValueError("Unknown data grid type")         
        
    assert len(data) == nrows and len(data[0]) == ncols

    # ------------- Create and setup figure
    fig, ax, pad, th = _setup_plotgrid(nrows,ncols,xlabel,ylabel,sharex,sharey,s)
    
    # Now we create the images
    images = []
    colorbars = []
    for i in range(nrows):
        imline = []
        cbline = []
        for j in range(ncols):
            im = ax[i][j].pcolormesh(X[i][j],Y[i][j],data[i][j], cmap = cmap, norm = norm, vmin = vmin, vmax = vmax)
            ax[i][j].set_box_aspect(1)
            # Scale
            ax[i][j].set_xscale(xscale)
            ax[i][j].set_yscale(yscale)
            cbar = add_nice_colorbar(im, 'top', pad=pad, thickness=th, label=label)
            imline.append(im)
            cbline.append(cbar)
            
        images.append(imline)
        colorbars.append(cbline)

    return fig, ax, images, colorbars

"""
