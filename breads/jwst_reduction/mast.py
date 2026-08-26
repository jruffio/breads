from astroquery.mast import MastMissions


def mast_search_observation_data( program, observation, file_suffix="uncal",
    download=False, download_path=None,
    **search_kwargs,
):
    """Search for JWST data products via MAST via Astroquery.

    MAST by default wants to give you high level data products; here we would in general
    rather get the original uncal files and process them ourselves.

    Parameters
    ----------
    program : str
        JWST program ID, e.g. '1282'
    observation : str
        JWST observation ID, e.g. '01'
    file_suffix : str
        The suffix of the files to return. Default is 'uncal', but could also be 'rate', 'cal', etc.
    search_kwargs : dict
        Additional search criteria to pass to the MAST search.
        These are passed directly to the Astroquery MastMissions.query_criteria() function, so
        see that documentation for details. Some common ones include e.g.
            grating = 'G395H'
            filter = 'F290LP'
            detector = 'NRS2'
    download : bool
        Whether to download the files found. Default is False.
    download_path : str or None
        The directory to download the files to. If None, uses default behavior for astroquery.mast which is
        to create within the current working directory a subdirectory called 'mastDownload' and download into that.

    Example Usage:
    -------------
    products = mast_search_observation_data(8063, 6, 'uncal', grating='G395H', detector='NRS2')

    Returns
    -------
     Either a list of Astroquery MAST product records, or if download=True, a Table of the downloaded files and statuses

    """

    # init search interface
    mission_search = MastMissions(mission="jwst")

    # handle special cases for search criteria
    if ( "grating" in search_kwargs ):  # the parameter for this is actually called 'nirspec_grating'
        search_kwargs["nirspec_grating"] = search_kwargs["grating"]
        del search_kwargs["grating"]

    # do the search
    exposures = mission_search.query_criteria(
        program=program, observtn=observation, **search_kwargs
    )

    print(f"Found {len(exposures)} exposures for that search")
    print("Finding all associated products in MAST... (this step can be slow)")
    products = mission_search.get_product_list(exposures)
    print(f"Found {len(products)} total associated data products")

    print(f"Filtering to {file_suffix} files")

    filtered = mission_search.filter_products( products, extension="fits",
        type="science", file_suffix=[ "_" + file_suffix ],  # note, the initial underscore is needed. Just to mess with us...
        )
    if "detector" in search_kwargs:
        # filter_products doesn't actually allow filtering on this, so do so manually ourselves
        detector_match = [ search_kwargs["detector"].lower() in row["filename"] for row in filtered ]
        filtered = filtered[detector_match]

    print(f"Filtered to {len(filtered)} matching data products")

    if download:
        print(f"Downloading {len(filtered)} files to {download_path}")
        return mission_search.download_products(filtered, download_dir=download_path, flat=True)
    else:
         return filtered
