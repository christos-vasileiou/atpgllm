from setuptools import setup, find_packages
import sys

# Metadata
PROJECT_NAME = "atpgllm" 
AUTHOR = "christos-vasileiou"
AUTHOR_EMAIL = "chrivasileiou2@gmail.com"
GITHUB_URL = f"https://github.com/{AUTHOR}/{PROJECT_NAME}"
DESCRIPTION = "LLM model that can run Automated Test Pattern Generation (ATPG) and Design Verification (DV) algorithm on synthesized verilog-based netlists."
LONG_DESCRIPTION = open("README.md").read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

# Read requirements while skipping comments and empty lines
with open("requirements-core.txt") as f:
    REQUIRED_PACKAGES = [
        line.strip() 
        for line in f
        if line.strip() and not line.startswith("#")
    ]

VERSION = "0.6.14"

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=GITHUB_URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "License :: Free For Educational Use",
        "Topic :: Scientific/Engineering",
        "Development Status :: 4 - Beta",
    ],
    python_requires=f">={sys.version.split()[0]}",
)

