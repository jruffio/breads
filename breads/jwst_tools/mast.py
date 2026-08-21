import os
from astroquery.mast import Observations



def download_one_file(filename, output_dir='./', verbose=False):
    """Download a single file from MAST, given the filename (e.g. 'jw01414014001_02101_00001_nrs2_uncal.fits')

    Parameters
    ----------
    filename : str
        The name of the file to download from MAST
    output_dir : str
        The directory to download the file into

    Returns
    -------
    str
        The path to the downloaded file

    """
    data_uri = f"mast:JWST/product/{filename}"

    # Download the file to your current working directory
    status, message, url = Observations.download_file(data_uri, local_path=output_dir)
    if status == 'COMPLETE':
        if verbose:
            print("Download complete")
        return os.path.join(output_dir, filename)
    else:
        raise RuntimeError(f"Error, download unsuccessful: {status} {message}")