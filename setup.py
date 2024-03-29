import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydex",
    version="0.0.9",
    author="Kennedy Putra Kusumo",
    author_email="kennedy.putra.kusumo@gmail.com",
    description="A Python package for optimal experiment design.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KennedyPutraKusumo/pydex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'numdifftools',
        'scipy',
        'cvxpy',
        'dill',
        'numpy',
        'corner',
        'emcee',
    ],
)
