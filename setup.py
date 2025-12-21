"""
Setup script for Bayesian Cognitive Field package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="bayesian_cognitive_field",
    version="0.1.0",
    author="Sam Leizerman",
    description="Bayesian recursive dynamics with fractional memory for cognitive tensor fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samleizerman/bayesian_cognitive_field",  # Update with actual URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
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
            "bcf-simple=bayesian_cognitive_field.examples.simple_evolution:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bayesian_cognitive_field": ["*.tex", "*.md"],
    },
)
