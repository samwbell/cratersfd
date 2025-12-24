from setuptools import setup, find_packages

setup(
    name="cratersfd",
    version="1.0.0",
    description="For analyzing crater size-frequency distributions.",
    author="Sam Bell",
    author_email="swbell11@gmail.com",
    url="https://github.com/samwbell/cratersfd",
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={"": ["LICENSE"]},
    install_requires=[
        "numpy>=1.26.4",
        "matplotlib>=3.9.1",
        "scipy>=1.14.0",
        "pandas>=2.2.2",
        "geopandas>=1.0.1",
        "shapely>=2.1.0",
        "pycrs>=1.0.2",
        "pyproj>=3.7.1",
        "ash @ git+https://github.com/ajdittmann/ash.git@master#egg=ash"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires='>=3.12',  # Minimum Python version required
)
