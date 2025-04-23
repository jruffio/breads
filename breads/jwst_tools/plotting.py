import os
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import astropy.io.fits as fits
import astropy.visualization

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
    """ Display a series of images, and save the result to a PDF.

    Parameters
    ----------
    filenames
    output_name

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



