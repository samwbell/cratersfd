from setuptools import setup, find_packages

with open("README_pypi.md", "r") as f:
    long_description = f.read()

setup(
    name="cratersfd",
    version="1.0.0",
    description="For analyzing crater size-frequency distributions.",
    author="Sam Bell",
    author_email="swbell11@gmail.com",
    url="https://github.com/samwbell/cratersfd",
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.14.0",
        "pandas>=2.2.2",
        "ash @ git+https://github.com/ajdittmann/ash.git@master#egg=ash"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    license_files=["LICENSE"],
    python_requires='>=3.12',  # Minimum Python version required
)
