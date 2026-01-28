"""
Setup script for Liorhybrid package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="Liorhybrid",
    version="0.1.0",
    author="Sam Leizerman",
    description="Bayesian recursive dynamics with fractional memory for cognitive tensor fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JewJitsu11B/Liorhybrid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.13",
    install_requires=[
        "torch>=2.5.0",
        "numpy>=2.1.0",
        "scipy>=1.14.0",
        "matplotlib>=3.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "liorhybrid=Liorhybrid.cli:main",
            "liorhybrid-train=Liorhybrid.cli:train_entrypoint",
            "liorhybrid-inference=Liorhybrid.cli:inference_entrypoint",
            "liorhybrid-simple=Liorhybrid.examples.simple_evolution:main",
        ],
    },
    include_package_data=True,
    package_data={
        "Liorhybrid": ["*.tex", "*.md"],
    },
)
