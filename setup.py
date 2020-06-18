import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kabl",
    version="2.0.0",
    author="Meteo-France",
    author_email="thomas.rieutord@meteo.fr",
    description="BL estimation from lidar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasRieutord/kabl",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
