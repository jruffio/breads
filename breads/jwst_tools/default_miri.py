import numpy as np

nodes_dict = {}

nodes_dict["1A"] = np.linspace(4.89, 5.75, 40, endpoint=True)
nodes_dict["1B"] = np.linspace(5.65, 6.64, 40, endpoint=True)
nodes_dict["1C"] = np.linspace(6.51, 7.67, 40, endpoint=True)
nodes_dict["2A"] = np.linspace(7.47, 8.80, 80, endpoint=True)
nodes_dict["2B"] = np.linspace(8.66, 10.14, 80, endpoint=True)
nodes_dict["2C"] = np.linspace(10.00, 11.712, 80, endpoint=True)
nodes_dict["3A"] = np.linspace(11.55, 13.47, 80, endpoint=True)
nodes_dict["3B"] = np.linspace(13.34, 15.56, 80, endpoint=True)
nodes_dict["3C"] = np.linspace(15.3, 18.1, 80, endpoint=True)

wv_nodes_3D_dict = {}

wv_nodes_3D_dict["1A"] = np.linspace(4.82, 5.79, 5, endpoint=True)
wv_nodes_3D_dict["1B"] = np.linspace(5.61, 6.69, 5, endpoint=True)
wv_nodes_3D_dict["1C"] = np.linspace(6.46, 7.72, 5, endpoint=True)
wv_nodes_3D_dict["2A"] = np.linspace(7.42, 8.82, 5, endpoint=True)
wv_nodes_3D_dict["2B"] = np.linspace(8.66, 10.14, 5, endpoint=True)
wv_nodes_3D_dict["2C"] = np.linspace(10.00, 11.712, 5, endpoint=True)
wv_nodes_3D_dict['3A'] = np.linspace(11.50, 13.60, 5, endpoint=True)
wv_nodes_3D_dict['3C'] = np.linspace(15.3, 18.1, 5, endpoint=True)


x_nodes_3D_5x = np.arange(-2, 2.0001, 0.05)
y_nodes_3D_5x = np.arange(-2, 2.0001, 0.05)


wv_sampling_dict = {}
wv_sampling_dict["1A"] = np.linspace(4.89, 5.75, 1050, endpoint=True)
wv_sampling_dict["1B"] = np.linspace(5.65, 6.64, 1050, endpoint=True)
wv_sampling_dict["1C"] = np.linspace(6.51, 7.67, 1050, endpoint=True)

wv_sampling_dict["2A"] = np.linspace(7.47, 8.80, 1050, endpoint=True)
wv_sampling_dict["2B"] = np.linspace(8.66, 10.14, 1050, endpoint=True)
wv_sampling_dict["2C"] = np.linspace(10.00, 11.712, 1050, endpoint=True)

wv_sampling_dict["3A"] = np.linspace(11.55, 13.47, 1050, endpoint=True)
wv_sampling_dict["3B"] = np.linspace(13.34, 15.56, 1050, endpoint=True)
wv_sampling_dict["3C"] = np.linspace(15.3, 18.1, 1050, endpoint=True)

wv_ref_dict = {"1A":5.4, "1B":6, "1C":7, "2A":8, "2B":9, "2C":11 ,"3A":12, "3B":14, "3C":16.5}

aperture_dict = {"1A":0.2, "1B":0.2, "1C":0.2, "2A":0.3, "2B":0.3, "2C":0.3, "3A":0.4, "3B":0.4, "3C":0.4}

fov_dict = {"1A":0.6, "1B":0.6, "1C":0.6, "2A":0.9, "2B":0.9, "2C":0.9, "3A":1.2, "3B":1.2, "3C":1.2}


def sub_band_from_name(sub_band_name):
    """Convert the BAND header value of a MIRI/MRS file to its sub-band letter: 'SHORT' -> 'A', 'MEDIUM' -> 'B', 'LONG' -> 'C'."""
    sub_bands = {'SHORT': 'A', 'MEDIUM': 'B', 'LONG': 'C'}
    if sub_band_name not in sub_bands:
        raise ValueError(f"Sub-band {sub_band_name}  is not recognized, must be SHORT, MEDIUM or LONG")
    return sub_bands[sub_band_name]
