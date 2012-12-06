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
from distutils.unixccompiler import UnixCCompiler

import commands

def pkgconfig(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])

    kw["libraries"].append("boost_python")
    return kw


#add here some options to handle additional compiler parameters
cliopts =[]
cliopts.append(("noforeach",None,"Set noforeach option for C++"))

op = FancyGetopt(option_table=cliopts)
args,opts = op.getopt()

include_dirs = []
library_dirs = []


try: include_dirs.append(opts.numpyincdir)
except:pass

#in the end we need to add the Python include directory
include_dirs.append(get_python_inc())

cc = new_compiler()
try:
    cc.set_executables(compiler_so = os.environ['CC'])
except:
    print "Environment variable CC not found!"

compile_args = ["-std=c++0x","-g","-O0"]
#now we try to compile the test code
try:
    print "run compiler test for nullptr ..."
    cc.compile(['ccheck/nullptr_check.cpp'],extra_preargs=compile_args)
    print "compiler supports nullptr - passed!"
except:
    print "no nullptr support!"
    compile_args.append("-Dnullptr=NULL")

try:
    print "run compiler check for foreach loops ..."
    cc.compile(['ccheck/foreach_check.cpp'],extra_preargs=compile_args)
    print "compiler supports foreach loops!"
except:
    print "no support for foreach loops!"
    compile_args.append("-DNOFOREACH")


files = ["src/cdma.cpp","src/Factory.cpp","src/GroupWrapper.cpp",
         "src/DatasetWrapper.cpp","src/DataItemWrapper.cpp",
         "src/Exceptions.cpp","src/WrapperHelpers.cpp",
         "src/AttributeWrapper.cpp","src/Selection.cpp",
         "src/DimensionWrapper.cpp","src/DimensionManager.cpp",
         "src/Types.cpp","src/TupleIterator.cpp"]

cdma = Extension("cdmacore",files,
                 extra_compile_args = compile_args,
                 **pkgconfig('cdmacore'))

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

