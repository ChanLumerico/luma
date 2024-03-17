import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="luma-ml",
    version='0.6.4',
    author="ChanLumerico",
    author_email="greensox284@gmail.com",
    description="A Comprehensive Python Module for Machine Learning and Data Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChanLumerico/LUMA",
    packages=setuptools.find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn'
    ]
)
