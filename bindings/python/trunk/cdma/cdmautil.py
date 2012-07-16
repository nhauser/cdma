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
# Created on: Jul 16, 2011
#     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
#


import cdmacore

def get_dimensions(group):
    """
    get_dimensions(grp):
    Filter function that returns a list of groups from a group object.

    arguments:
    group .......... group from which to pick dimension objects

    return:
    List with Dimension objects
    """

    l = []

    for c in group.childs:
        if isinstance(c,cdmacore.Dimension): l.append(c)

    return l

def get_groups(group):
    """
    get_groups(group):
    Filter function returning a list of groups stored below a group object.

    arguments:
    group ............ group object where to search for child groups

    return:
    list of child groups of group
    """

    l = []

    for g in group.childs:
        if isinstance(g,cdmacore.Group): l.append(g)

    return l

def get_dataitems(group):
    """
    get_dataitems(group):
    filter function returning a list of data items stored below group.

    arguments:
    group .............. group object where to search for data items

    return:
    list of data item childs of group
    """

    l = []

    for d in group.childs:
        if isinstance(d,cdmacore.DataItem): l.append(d)

    return l
