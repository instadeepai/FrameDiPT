"""Setup module."""
from setuptools import setup

setup(
    name="framedipt",
    version="0.1.0",
    description="Diffusion Model for Protein Structure Inpainting",
    packages=["framedipt", "openfold", "experiments", "evaluation"],
    package_dir={
        "framedipt": "./framedipt",
        "openfold": "./openfold",
        "experiments": "./experiments",
        "evaluation": "./evaluation",
    },
)
