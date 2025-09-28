# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "optimizer.cores.optimizer_core",
        sources=["src/optimizer/cores/optimizer_core.pyx"],  # или .c если уже сгенерировано
        include_dirs=[numpy.get_include()],
        language="c",
    ),
    Extension(
        "optimizer.cores.overbooking_core",
        sources=["src/optimizer/cores/overbooking_core.pyx"],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
]

setup(
    name="optimizer",
    version="0.1.0",
    package_dir={"": "src"},
    packages=["optimizer", "optimizer.cores"],  # или используйте find_packages
    ext_modules=cythonize(extensions, language_level="3"),
)