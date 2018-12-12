#!/usr/bin/env python

import os
import sys
import glob

from setuptools import setup, Extension

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

import tensorflow as tf  # NOQA

sys.path.append("celeriteflow")

from cpp_extension import BuildExtension, CppExtension  # NOQA


path = os.path.join("celeriteflow", "ops")
cpp_files = glob.glob(os.path.join(path, "*.cc"))

flags = ["-O2"]
if sys.platform == "darwin":
    flags += ["-mmacosx-version-min=10.9"]

extensions = [CppExtension(
    "celeriteflow.celerite_op",
    cpp_files,
    include_dirs=[path],
    extra_compile_args=flags,
    extra_link_args=flags,
)]

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
    ext_modules=extensions,
    description="",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["tensorflow"],
    # classifiers=[
    #     "Development Status :: 5 - Production/Stable",
    #     "Intended Audience :: Developers",
    #     "Intended Audience :: Science/Research",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    #     "Programming Language :: Python",
    # ],
)
