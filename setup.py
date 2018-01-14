#!/usr/bin/env python

import os
import tempfile

import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def has_library(compiler, libname):
    """Return a boolean indicating whether a library is found."""
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as srcfile:
        srcfile.write("int main (int argc, char **argv) { return 0; }")
        srcfile.flush()
        outfn = srcfile.name + ".so"
        try:
            compiler.link_executable(
                [srcfile.name],
                outfn,
                libraries=[libname],
            )
        except setuptools.distutils.errors.LinkError:
            return False
        if not os.path.exists(outfn):
            return False
        os.remove(outfn)
    return True


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class build_ext(_build_ext):
    c_opts = {
        "msvc": ["/EHsc", "/DNODEBUG"],
        "unix": ["-DNODEBUG"],
    }

    def build_extensions(self):
        import tensorflow as tf
        include_dirs = [tf.sysconfig.get_include()]
        include_dirs.append(os.path.join(
            include_dirs[0], "external/nsync/public"))

        for ext in self.extensions:
            ext.include_dirs = include_dirs + ext.include_dirs

        # Compiler flags
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts.append("-DVERSION_INFO=\"{0:s}\""
                    .format(self.distribution.get_version()))
        print("testing C++14/C++11 support")
        opts.append(cpp_flag(self.compiler))

        flags = ["-stdlib=libc++", "-funroll-loops",
                 "-Wno-unused-function", "-Wno-uninitialized",
                 "-Wno-unused-local-typedefs"]
        libraries = ["m", "c++"]

        for ext in self.extensions:
            ext.library_dirs += [tf.sysconfig.get_lib()]

        # Mac specific flags and libraries
        if sys.platform == "darwin":
            flags += ["-march=native", "-mmacosx-version-min=10.9"]
            for lib in libraries:
                for ext in self.extensions:
                    ext.libraries.append(lib)
            for ext in self.extensions:
                ext.extra_link_args += ["-mmacosx-version-min=10.9",
                                        "-march=native"]
        else:
            libraries += ["stdc++"]
            for lib in libraries:
                if not has_library(self.compiler, lib):
                    continue
                for ext in self.extensions:
                    ext.libraries.append(lib)

        # Check the flags
        print("testing compiler flags")
        for flag in flags:
            if has_flag(self.compiler, flag):
                opts.append(flag)

        for ext in self.extensions:
            ext.extra_compile_args = opts

        # Run the standard build procedure.
        _build_ext.build_extensions(self)


if __name__ == "__main__":
    import sys

    # Publish the library to PyPI.
    if "publish" in sys.argv[-1]:
        os.system("python setup.py sdist upload")
        sys.exit()

    fmt_filename = lambda fn: os.path.join("celeriteflow", "ops", fn)
    extensions = [
        Extension("celeriteflow.ops.celerite_op",
                  sources=[
                      fmt_filename("celerite_factor_op.cc"),
                      fmt_filename("celerite_factor_grad_op.cc"),
                  ],
                  language="c++"),
    ]

    # Hackishly inject a constant into builtins to enable importing of the
    # package before the library is built.
    # if sys.version_info[0] < 3:
    #     import __builtin__ as builtins
    # else:
    #     import builtins
    # builtins.__CELERITEFLOW_SETUP__ = True
    # import

    setup(
        name="celeriteflow",
        # version=george.__version__,
        author="Daniel Foreman-Mackey",
        author_email="foreman.mackey@gmail.com",
        # url="https://github.com/dfm/george",
        # license="MIT",
        packages=["celeriteflow", "celeriteflow.ops"],
        ext_modules=extensions,
        # description="Blazingly fast Gaussian Processes for regression.",
        # long_description=open("README.rst").read(),
        # package_data={"": ["README.rst", "LICENSE", "AUTHORS.rst",
        #                    "HISTORY.rst"]},
        # install_requires=["numpy", "scipy", "pybind11"],
        # include_package_data=True,
        cmdclass=dict(build_ext=build_ext),
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
