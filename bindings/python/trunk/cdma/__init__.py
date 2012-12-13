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

from config import default_plugin_path
from cdmacore import _factory
from cdmacore import Group
from cdmacore import Dataset
from cdmacore import DataItem
from cdmacore import Dimension
from cdmautil import get_dimensions
from cdmautil import get_groups
from cdmautil import get_dataitems

#need to add some code here to evaluate the CDMA_PLUGIN_PATH environment 
#variable

#need to add some code here to evaluate the CDMA_DICTIONARY_PATH environment
#variable

#here we need most probably a more sophisticated configuration facility
_factory.init(default_plugin_path)
print _factory


def open_dataset(path):
    return _factory.open_dataset(path)

#cleanup function should be called when the python interpreter exits
import atexit
atexit.register(_factory.cleanup)
