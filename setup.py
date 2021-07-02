from setuptools import setup, find_packages
from pkg_resources import parse_requirements

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('./requirements.txt') as requirements:
    setup(
        name='breads',
        version='0.0.1',
        author="Shubh Agrawal, Jean-Baptiste Ruffio",
        author_email="shubh@caltech.edu, jruffio@caltech.edu",
        description="Broad Respository for Exoplanet Analysis, Discovery, and Spectroscopy",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jruffio/breads",
        packages=find_packages(),
        install_requires=requirements.readlines(),
        include_package_data=True,
        package_data={'': ['data/*']},
    )