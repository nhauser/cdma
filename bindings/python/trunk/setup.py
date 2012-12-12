#
# (c) Copyright 2012 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
#
# This file is part of cdma-python.
#
# cdma-python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# cdma-python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
# ***********************************************************************
#
# Created on: Jun 26, 2011
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#

#setup script for libpninx-python package
import sys
import os
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from distutils.fancy_getopt import FancyGetopt
from distutils.fancy_getopt import fancy_getopt
from distutils.ccompiler import new_compiler
from distutils.sysconfig import get_config_vars
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils import misc_util

#get rid of this stupid warning for a compiler option never needed - everything
#else remains the same
(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
            flag for flag in opt.split() if flag != '-Wstrict-prototypes'
            )

import commands

cliopts =[]
cliopts.append(("h5libdir=",None,"HDF5 library path"))
cliopts.append(("h5incdir=",None,"HDF5 include path"))
cliopts.append(("h5libname=",None,"HDF5 library name"))
cliopts.append(("nxlibdir=",None,"PNI NX library path"))
cliopts.append(("nxincdir=",None,"PNI NX include path"))
cliopts.append(("utlibdir=",None,"PNI utilities library path"))
cliopts.append(("utincdir=",None,"PNI utilities include path"))
cliopts.append(("numpyincdir=",None,"Numpy include path"))
cliopts.append(("noforeach",None,"Set noforeach option for C++"))
cliopts.append(("debug",None,"append debuging options"))

op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

debug = False
for o,v in op.get_option_order():
    if o == "debug":
        debug = True

def pkgconfig(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])

    kw["libraries"].append("boost_python")
   
    if not kw.has_key("extra_compile_args"):
        kw["extra_compile_args"] = []

    kw["include_dirs"].append(misc_util.get_numpy_include_dirs()[0])
    #kw["extra_compile_args"].append('-std=c++0x')

    if True:
        
        kw["extra_compile_args"].append('-O0')
        kw["extra_compile_args"].append('-g')
    return kw



files = ["src/cdma.cpp","src/Factory.cpp","src/GroupWrapper.cpp",
         "src/DatasetWrapper.cpp","src/DataItemWrapper.cpp",
         "src/Exceptions.cpp","src/WrapperHelpers.cpp",
         "src/AttributeWrapper.cpp","src/Selection.cpp",
         "src/DimensionWrapper.cpp","src/DimensionManager.cpp",
         "src/Types.cpp","src/TupleIterator.cpp"]

cdma = Extension("cdmacore",files,
                 language="c++",**pkgconfig('cdmacore'))

setup(name="cdma-python",
        author="Eugen Wintersberger",
        author_email="eugen.wintersberger@desy.de",
        description="Python bindings for CDMA ",
        long_description="""Python bindings for the CDMA C++ framework. Supports"
        actually only raw data access. Support for dictionaries will follow
        later.""",
        version = "0.1.0",
        ext_package="cdma",
        ext_modules=[cdma],
        packages = ["cdma"],
        license="GPL V2",

        )

