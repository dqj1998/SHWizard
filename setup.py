from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="shwizard",
    version="0.1.0",
    author="Qj D",
    author_email="dqj1998@gmail.com",
    description="A shell with AI wizard - Natural language to shell commands",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dqj1998/SHWizard",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "shwizard": ["data/*.yaml", "data/*.json"],
    },
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "prompt-toolkit>=3.0.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "platformdirs>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "build": [
            "pyinstaller>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "shwizard=shwizard.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
)
