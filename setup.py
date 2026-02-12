"""
Setup configuration for Scholar's Vault
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="scholars-vault",
    version="0.1.0",
    description="Agentic RAG system for academic research and knowledge management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Scholar's Vault Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=1.2.3",
        "langchain-core>=1.2.7",
        "langchain-text-splitters>=1.1.0",
        "pymupdf>=1.26.6",
        "pymupdf4llm>=0.2.9",
        "python-docx",
        "ebooklib",
        "beautifulsoup4",
        "qdrant-client>=1.16.2",
        "fastembed>=0.7.4",
        "torch>=2.0.0",
        "loguru>=0.7.3",
        "typer>=0.21.1",
        "rich>=14.2.0",
        "pyyaml>=6.0.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "scholars-vault=cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
