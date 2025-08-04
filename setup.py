"""
Self-Documenting PhD Notebook
An Obsidian-compatible research notebook with AI-powered automation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="self-documenting-phd-notebook",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@example.com",
    description="Obsidian-compatible research notebook with AI-powered automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/Self-Documenting-PhD-Notebook",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing :: Markup :: Markdown",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "rich>=12.0.0",
        "aiofiles>=0.8.0",
        "httpx>=0.24.0",
        "jinja2>=3.0.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.0",
        "semantic-version>=2.10.0",
        "gitpython>=3.1.0",
        "watchdog>=2.1.0",
    ],
    extras_require={
        "ai": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
            "tiktoken>=0.4.0",
            "sentence-transformers>=2.2.0",
            "chromadb>=0.4.0",
        ],
        "integrations": [
            "slack-sdk>=3.19.0",
            "google-api-python-client>=2.0.0",
            "dropbox>=11.36.0",
            "requests>=2.28.0",
            "beautifulsoup4>=4.11.0",
            "feedparser>=6.0.0",
        ],
        "science": [
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
            "scipy>=1.8.0",
            "plotly>=5.10.0",
            "seaborn>=0.11.0",
        ],
        "latex": [
            "pylatex>=1.4.0",
            "bibtexparser>=1.4.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
            "tiktoken>=0.4.0",
            "sentence-transformers>=2.2.0",
            "chromadb>=0.4.0",
            "slack-sdk>=3.19.0",
            "google-api-python-client>=2.0.0",
            "dropbox>=11.36.0",
            "requests>=2.28.0",
            "beautifulsoup4>=4.11.0",
            "feedparser>=6.0.0",
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
            "scipy>=1.8.0",
            "plotly>=5.10.0",
            "seaborn>=0.11.0",
            "pylatex>=1.4.0",
            "bibtexparser>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sdpn=phd_notebook.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "phd_notebook": [
            "templates/*.md",
            "templates/*.tex",
            "config/*.yaml",
            "schemas/*.json",
        ],
    },
)