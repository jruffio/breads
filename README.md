# breads
Broad Repository for Exoplanet Analysis, Discovery, and Spectroscopy

See the documentation:
https://breads.readthedocs.io/en/latest/

Installing breads for JWST on a Linux server using a conda environment (only tested on a single machine):

conda create -n jwst python==3.11.0
conda activate jwst
pip install jwst==1.12.5
pip install webbpsf==1.2.1
pip install PyAstronomy==0.20.0
pip install h5py==3.10.0
pip install pandas==2.1.4
pip install py==1.11.0
pip install PyQt5
