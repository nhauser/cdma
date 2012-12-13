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
cliopts.append(("boostlibdir=",None,"BOOST library path"))
cliopts.append(("boostincdir=",None,"BOOST header path"))
cliopts.append(("with-debug",None,"append debuging options"))
cliopts.append(("with-cpp11",None,"add C++11 support"))
cliopts.append(("plugin-path",None,"sets the default plugin search path"))


op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

debug = False
cpp_11_support = False
default_plugin_path = "/usr/lib/cdma/plugins"
for o,v in op.get_option_order():
    if o == "with-debug":
        debug = True

    if o == "boostlibdir":
        boost_library_dir = v

    if o == "boostincdir":
        boost_inc_dir = v

    if o == "with-cpp11":
        cpp_11_support = True

    if o == "plugin-path":
        default_plugin_path = v

def pkgconfig(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])

    try:
        kw["libraries"].append("boost_python")
    except:
        kw["libraries"] = ["boost_python"]

    try:
        kw["library_dirs"].append(boost_library_dir)
    except:
        pass

    try:
        kw["include_dirs"].append(boost_inc_dir)
    except:
        pass

   
    if not kw.has_key("extra_compile_args"):
        kw["extra_compile_args"] = []

    kw["include_dirs"].append(misc_util.get_numpy_include_dirs()[0])
    #kw["extra_compile_args"].append('-std=c++0x')

    if cpp_11_support:
        kw["extra_compile_args"].append("-std=c++0x")

    if debug:
        kw["extra_compile_args"].append('-O0')
        kw["extra_compile_args"].append('-g')

    return kw

#create the configuration file
config = open("cdma/config.py","w")
config.write("default_plugin_path = \"%s\"\n" %(default_plugin_path))
config.close()



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
        script_args = args
        )

