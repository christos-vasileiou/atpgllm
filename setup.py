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
REQUIRED_PACKAGES = [line for line in open("requirements.txt").readlines() if 'https://' not in line]
REQUIRED_LINKS = [line.strip().split()[-1] for line in open("requirements.txt").readlines() if 'https://' in line]
VERSION = "1.0.1"


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
  #dependency_links=REQUIRED_LINKS,
  classifiers=[
    # Choose your license from the "License" classifiers list:
    # https://pypi.org/classifiers/
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "License :: Free For Educational Use",
    "Topic :: Scientific/Engineering",
    "Development Status :: 4 - Beta",
  ],
  python_requires=f">={sys.version.split()[0]}",
)

