/*
 * (c) Copyright 2012 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of cdma-python.
 *
 * cdma-python is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * cdma-python is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Created on: Jun 26, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#include "GroupWrapper.hpp"
#include "DataItemWrapper.hpp"
#include "WrapperHelpers.hpp"

#include <cdma/navigation/IContainer.h>

//================implementation of class methods==============================
object GroupWrapper::__getitem__(const std::string &path) const
{
    //check groups
    for(auto v: ptr()->getGroupList())
        if(path == v->getShortName()) return object(new GroupWrapper(v));

    //check data items
    for(auto v: ptr()->getDataItemList())
        if(path == v->getShortName()) return object(new DataItemWrapper(v));


    //throw an exception here
}

//----------------------------------------------------------------------------
GroupWrapper GroupWrapper::parent() const
{
    return GroupWrapper(ptr()->getParent());
}

//----------------------------------------------------------------------------
GroupWrapper GroupWrapper::root() const
{
    return GroupWrapper(ptr()->getRoot());
}

//----------------------------------------------------------------------------
tuple GroupWrapper::childs() const
{
    list l;
    
    for(auto v: ptr()->getGroupList()) l.append(GroupWrapper(v));
    for(auto v: ptr()->getDataItemList()) l.append(DataItemWrapper(v));

    return tuple(l);
}

//----------------------------------------------------------------------------
tuple GroupWrapper::groups() const
{
    list l;
    for(auto v:ptr()->getGroupList()) l.append(GroupWrapper(v));
    return tuple(l);
}

//----------------------------------------------------------------------------
tuple GroupWrapper::items() const
{
    list l;
    for(auto v:ptr()->getDataItemList()) l.append(DataItemWrapper(v));
    return tuple(l);
}

//----------------------------------------------------------------------------
std::list<IDimensionPtr> GroupWrapper::dimensions() const
{
    return ptr()->getDimensionList();
}

//====================helper function to create python class==================
void wrap_group()
{

    wrap_container<IGroupPtr>("GroupContainer");
    wrap_attribute_manager<IGroupPtr>("GroupAttributeManager");

    class_<GroupWrapper,bases<ContainerWrapper<IGroupPtr>> >("Group")
        .def_readwrite("attrs",&GroupWrapper::attrs)
        .add_property("parent",&GroupWrapper::parent)
        .add_property("root",&GroupWrapper::root)
        .add_property("childs",&GroupWrapper::childs)
        .add_property("items",&GroupWrapper::items)
        .add_property("gruops",&GroupWrapper::groups)
        .add_property("dims",&__dimensions__<GroupWrapper>)
        .def("__getitem__",&GroupWrapper::__getitem__)
        ;
}


