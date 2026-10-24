import os
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import astropy.io.fits as fits
import astropy.visualization
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri

# Various functions for plotting and displaying JWST images,
# for instance to check the results of reductions and analyses


###########################################################################
# Displaying and plotting 2D images from JWST pipeline reductions and other processing

def _default_plot_output_name(data_filename, plot_label='image', output_dir='./'):
    """ Supply a reasonable default filename for plot PDF output

    Parameters
    ----------
    data_filename : str
        Some FITS filename, to read header metadata from
    plot_label : str
        Descriptive label to put into the filename

    Returns
    -------

    """
    hdr = fits.getheader(data_filename)
    visit_id = hdr['VISIT_ID']
    detector = hdr['DETECTOR'].lower()
    return os.path.join(output_dir, f'plots_{plot_label}_jw{visit_id}_{detector}.pdf')


def plot_2d_image(filename, ax=None, extname='SCI', colorbar=True):
    """Display function for one 2D image, e.g. rate or cal file
    Plots the image, with hopefully-reasonable default scaling on an asinh stretch

    Parameters
    ----------
    filename : str
        Input image to displau
    ax : matplotlib.Axes instance or None
        Axes to display into. If None, the current Axes will be used
    extname : str
        FITS extension name to display
    colorbar : bool
        Show a color bar for the figure?

    Returns
    -------

    """

    if ax is None:
        ax = plt.gca()

    with fits.open(filename) as hdul:
        im = hdul[extname].data

    norm = astropy.visualization.simple_norm(im, stretch='asinh', min_percent=1, max_percent=99, asinh_a=0.01 )
    ax.imshow(im, norm=norm)
    if colorbar:
        plt.colorbar(mappable=ax.images[0], ax=ax)
    ax.set_title(os.path.basename(filename))

    plt.tight_layout()


def plot_2d_image_set(filenames, output_dir='./', plot_label='plots', output_name=None, suptitle=None):
    """
    Display a series of images, and save the result to a PDF.

    Parameters
    ----------
    filenames : list of str
        Filenames for images to display
    output_name : str
        Filename for output file

    Returns
    -------

    """
    if output_name is None:
        output_name = _default_plot_output_name(filenames[0], output_dir=output_dir, plot_label=plot_label)

    with PdfPages(output_name) as pdf:

        for fn in filenames:
            fig, ax = plt.subplots(figsize=(16, 9))
            plot_2d_image(fn, ax=ax)
            if suptitle:
                fig.suptitle(suptitle, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 0.95, 0.97])  # leave space at top for suptitle
            pdf.savefig(fig)
            plt.close(fig)

    print("Plots saved to "+output_name)

def plot_2d_image_sets_side_by_side(filenames1, filenames2, output_dir='./', plot_label='plots', output_name=None,
                                    suptitle=None):
    """ Display a series of paired images, and save the result to a PDF.

    Parameters
    ----------
    filenames1 : list of str
        Filenames for left hand side
    filenames2 : list of str
        Filenames for right hand side
    output_name : str
        Filename for output file

    Returns
    -------

    """
    if output_name is None:
        output_name = _default_plot_output_name(filenames1[0], output_dir=output_dir, plot_label=plot_label)

    with PdfPages(output_name) as pdf:

        for fn1, fn2 in zip(filenames1, filenames2):
            fig, axes = plt.subplots(figsize=(16, 9), ncols=2)
            plot_2d_image(fn1, ax=axes[0])
            plot_2d_image(fn2, ax=axes[1])
            axes[1].images[0].norm = axes[0].images[0].norm # make sure images have same stretch
            if suptitle:
                fig.suptitle(suptitle, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 0.95, 0.97])  # leave space at top for suptitle
            pdf.savefig(fig)
            plt.close(fig)

    print("Plots saved to "+output_name)


def save_cube_as_gif(cube, filename="cube.gif", fps=10, cmap="viridis", vmin=None, vmax=None,extent=None,wv_nodes=None,dpi=100):
    if vmin is None:
        vmin = np.nanmin(cube)
    if vmax is None:
        vmax = np.nanmax(cube)

    fig, ax = plt.subplots()
    im = ax.imshow(cube[0], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",extent=extent,aspect='equal')
    plt.colorbar(im, ax=ax)
    if wv_nodes is not None:
        title = ax.set_title("Frame wv={0:.3f}".format(wv_nodes[0]))
    else:
        title = ax.set_title("Frame 0")


    def update(i):
        im.set_data(cube[i])
        if wv_nodes is not None:
            title.set_text("Frame wv={0:.3f}".format(wv_nodes[i]))
        else:
            title.set_text(f"Frame {i}")
        return im, title

    ani = FuncAnimation(fig, update, frames=cube.shape[0], interval=1000/fps, blit=True)
    ani.save(filename, writer="pillow", fps=fps, dpi=dpi)
    plt.close()


def point_cloud_interpolator_2d(x,y,wv_sampling, data,bad_pixels, wv0):
        """
        Generate a 2D point cloud interpolator at a given wavelength.

        Parameters
        ----------
        x : ndarray
            x coordinates of the point cloud. Same shape as data.
        y : ndarray
            y coordinates of the point cloud. Same shape as data.
        wv_sampling : ndarray
            Wavelength sampling array. Corresponds to the 2nd dimension of data, ie columns.
        data : ndarray
            Data. Shape should be (N_rows, N_wavelengths).
        bad_pixels : ndarray
            Array indicating bad pixels. np.nan for bad, 1 for good.
        wv0 : float
            Wavelength slice at which to interpolate. Since the wavelength sampling is discrete, the function will just pick the closest wavelength sample.

        Returns
        -------
        pointcloud_interp : scipy.interpolate.LinearTriInterpolator
            A 2D interpolator object that can be used to evaluate the interpolated data at any (x,y) position.

        """
        if wv0 is None:
            wv0 = np.nanmedian(wv_sampling)

        wv0_index = np.argmin(np.abs(wv_sampling - wv0))

        where_good = np.where(np.isfinite(bad_pixels[:, wv0_index])*np.isfinite(x[:, wv0_index])*np.isfinite(y[:, wv0_index])*np.isfinite(data[:, wv0_index]))
        if np.size(where_good[0]) <3:
            return None
        x = x[where_good[0], wv0_index]
        y = y[where_good[0], wv0_index]
        z = data[where_good[0], wv0_index]
        filtered_triangles = filter_big_triangles(x, y, 0.2)
        # Create filtered triangulation
        filtered_tri = tri.Triangulation(x, y, triangles=filtered_triangles)
        # Perform LinearTriInterpolator for filtered triangulation
        pointcloud_interp = tri.LinearTriInterpolator(filtered_tri, z)

        return pointcloud_interp


def filter_big_triangles(X,Y, max_edge_length):
    """ Create a triangulation of X, Y points, and filter based on edge length

    Parameters
    ----------
    X
    Y
    max_edge_length

    Returns
    -------

    """
    points = np.array([X,Y]).T
    # Create triangulation
    triangulation = tri.Triangulation(points[:, 0], points[:, 1])

    # Calculate triangle edge lengths
    edge_lengths = np.linalg.norm(
        points[triangulation.triangles[:, [0, 1, 2, 0]], :] - points[triangulation.triangles[:, [1, 2, 0, 1]], :],
        axis=2)

    # Check maximum edge length constraint
    valid_triangles = np.all(edge_lengths <= max_edge_length, axis=1)

    # Filter out sliver triangles
    filtered_triangles = triangulation.triangles[valid_triangles]

    return filtered_triangles

