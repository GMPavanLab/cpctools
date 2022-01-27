import setuptools

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SOAPify-GMPavanLab",
    version="0.0.2",
    author="Daniele Rapetti",
    author_email="daniele.rapetti@polito.it",
    description="A package for creating and studying SOAP fingerprints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GMPavanLab/SOAPify",
    project_urls={
        "Bug Tracker": "https://github.com/GMPavanLab/SOAPify/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",  # due to MDA
)
