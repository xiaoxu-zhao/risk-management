"""
Setup configuration for the Credit Risk Management toolkit.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="credit-risk-management",
    version="1.0.0",
    author="xiaoxu(Ivan) zhao",
    author_email="xiaoxu.zhao@gu.se",
    description="A comprehensive credit risk management toolkit for job-seeking demonstration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiaoxu-zhao/risk-management",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "jupyterlab>=4.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "risk-demo=src.data_loader:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="credit risk, machine learning, finance, basel, var, quantitative finance",
    project_urls={
        "Bug Reports": "https://github.com/xiaoxu-zhao/risk-management/issues",
        "Source": "https://github.com/xiaoxu-zhao/risk-management",
        "Documentation": "https://github.com/xiaoxu-zhao/risk-management/blob/main/README.md",
    },
)