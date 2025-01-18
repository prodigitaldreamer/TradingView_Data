from setuptools import setup, Extension
import pybind11
import sys

# Optional compiler args:
cpp_args = ['-std=c++11', '-O3']  # or c++14, c++17, etc. -O3 for optimization

ext_modules = [
    Extension(
        'mistaken_logic_ext',              # name of the resulting .so/.pyd
        sources=['mistaken_logic.cpp'],    # our .cpp file
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args
    ),
]

setup(
    name='mistaken_logic_ext',
    version='0.1.0',
    author='Me',
    description='C++ extension for pivot stack logic using Pybind11',
    ext_modules=ext_modules,
    zip_safe=False,
)