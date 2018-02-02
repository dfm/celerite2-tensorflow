#!/usr/bin/env python

import os
import sys

from setuptools import setup, Extension

import tensorflow as tf

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

fmt_filename = lambda fn: os.path.join("celeriteflow", "ops", fn)  # NOQA
ext = Extension(
    "celeriteflow.ops.celerite_op",
    sources=[
        fmt_filename("celerite_factor_op.cc"),
        fmt_filename("celerite_factor_grad_op.cc"),
        fmt_filename("celerite_solve_op.cc"),
        fmt_filename("celerite_solve_grad_op.cc"),
    ],
    language="c++"
)

# Includes
include_dirs = [tf.sysconfig.get_include()]
include_dirs.append(os.path.join(
    include_dirs[0], "external/nsync/public"))
include_dirs.append(os.path.join("celeriteflow", "ops"))
ext.include_dirs = include_dirs + ext.include_dirs

# Library
ext.library_dirs += [tf.sysconfig.get_lib()]
ext.libraries += ["m", "c++"]

# Flags
ext.extra_compile_args += ["-std=c++11", "-O2"]
if sys.platform == "darwin":
    ext.extra_compile_args += [
        "-march=native", "-mmacosx-version-min=10.9"]
    ext.extra_link_args += [
        "-march=native", "-mmacosx-version-min=10.9"]
else:
    ext.libraries += ["stdc++"]

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__CELERITEFLOW_SETUP__ = True
import celeriteflow  # NOQA

setup(
    name="celeriteflow",
    version=celeriteflow.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/celeriteflow",
    license="MIT",
    packages=["celeriteflow", "celeriteflow.ops"],
    ext_modules=[ext],
    description="",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE"]},
    include_package_data=True,
    install_requires=["tensorflow"],
    # classifiers=[
    #     "Development Status :: 5 - Production/Stable",
    #     "Intended Audience :: Developers",
    #     "Intended Audience :: Science/Research",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    #     "Programming Language :: Python",
    # ],
    zip_safe=True,
)
