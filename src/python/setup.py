from setuptools import setup

setup(
    name="GraphEMD",
    version="0.1.0",
    url="https://github.com/agusriscos/GraphEMD",
    license="Apache-2.0",
    author="agusriscos",
    author_email="agusriscos@gmail.com",
    description=(
        "Financial time series analysis using EMD/CEEMDAN, "
        "IMF dimensionality reduction, and graph transformation."
    ),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "pandas-datareader>=0.10.0",
        "plotly>=5.18.0",
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "torch-geometric>=2.3.0",
    ],
    packages=[
        "GraphEMD",
        "GraphEMD.data",
        "GraphEMD.conf",
        "GraphEMD.model",
        "GraphEMD.utils",
        "CommonUtils",
        "CommonUtils.data",
    ],
)
