from pathlib import Path
import matplotlib.pyplot as plt

# Paths
module_root = Path(__file__).parent
style_root = module_root / "styles"

# ---------------------- Methods to set Matplotlib style & enable/disable features

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

# ---------------------- General Utility Methods

def add_redshift_axis(ax, zlist, label = 'Redshift'):
    a2z = lambda a : 1./a-1.
    z2a = lambda z : 1./(z-1)
    secx = ax.secondary_xaxis('top', functions=(a2z, z2a), xlabel=label)
    secx.set_xticks(zlist)
    ax.tick_params('x', reset=1, top=0)
    #secx.set_xticklabels([f'{x}' for x in secx.get_xticks()])
    return secx
    
def add_nice_colorbar(im, location = 'right', label = None,
                     pad = 0.01, thickness = 0.03):

    ax = im.axes
    fig = ax.figure
    ax.set_aspect('equal', 'box')
    
    if location == 'right':
        cax = fig.add_axes([ax.get_position().x1+pad,ax.get_position().y0, thickness,ax.get_position().height])
    else:
        cax = fig.add_axes([ax.get_position().x0,ax.get_position().y1+pad,ax.get_position().width, thickness])
    cax.tick_params(axis='both',direction='out',reset=True)
    
    cbar = plt.colorbar(im, cax = cax, location=location, label=label)
    return cbar

# ---------------------- Full Figure Methods

"""Wrapper for the Matplotlib imshow function, for better looking image plots
"""
def imshow(data, xlabel = None, ylabel = None, cmap = None, label = None, norm = None, interp = None, cbar_loc = 'right', s = 1.0):
    # The s argument is used to globally adjust the figure size
    # Figure Size
    H = 3.5 * s
    L = 2.8 * s
    
    # Colorbar Placement & Size
    pad = 0.02
    th = 0.05
    
    fig, ax = plt.subplots(1,1, figsize=(L,H))
    
    # Add x/y labels if necessary
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Now we create the image
    im = ax.imshow(data, cmap = cmap, norm = norm, interpolation=interp, origin='lower')
    cbar = add_nice_colorbar(im, cbar_loc, pad=pad, thickness=th, label=label)

    return fig, ax, im, cbar
    
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
def imshow_grid(nrows, ncols, data,
                xlabel = None, ylabel = None, sharex = False, sharey = False,
                cmaps = None, labels = None, norms = None, interp = None,
                s = 1.0):

    # ------------- Check Data size
    # By default, we assume that data & co are 2D lists of numpy arrays.
    if not isinstance(data,list):
        # Its also possible to pass a 4D numpy array
        if len(data.shape) == 4:
            pass
        # Or a single array, for nrows = ncols = 1. Need to adapt in this case
        elif len(data.shape) == 2:
            data = [[data]]
            cmaps = [[cmaps]]
            labels = [[labels]]
        else:
            raise ValueError("Unknown data grid type")         
        
    assert len(data) == nrows and len(data[0]) == ncols

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

    # Add x/y labels if necessary
    for i in range(nrows):
        for j in range(ncols):
            if not (sharex and i != nrows - 1):
                ax[i,j].set_xlabel(xlabel)
            if not (sharey and j != 0):
                ax[i,j].set_ylabel(ylabel)

    # Now we create the images
    images = []
    colorbars = []
    for i in range(nrows):
        imline = []
        cbline = []
        for j in range(ncols):
            cmap = None
            label = None
            norm = None
            if cmaps is not None:
                cmap = cmaps[i][j]
            if labels is not None:
                label = labels[i][j]
            if norms is not None:
                norm = norms[i][j]
            im = ax[i][j].imshow(data[i][j], cmap = cmap, norm=norm, interpolation = interp, origin='lower')
            cbar = add_nice_colorbar(im, 'top', pad=pad, thickness=th, label=label)
            
            imline.append(im)
            cbline.append(cbar)
            
        images.append(imline)
        colorbars.append(cbline)

    return fig, ax, images, colorbars