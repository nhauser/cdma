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

#include <sstream>

#include "GroupWrapper.hpp"
#include "DataItemWrapper.hpp"
#include "WrapperHelpers.hpp"
#include "Exceptions.hpp"

#include <cdma/navigation/IContainer.h>

//================implementation of class methods==============================
object GroupWrapper::__getitem__(const std::string &name) const
{
    //check groups
    for(auto v: ptr()->getGroupList())
        if(name == v->getShortName()) return object(new GroupWrapper(v));

    //check data items
    for(auto v: ptr()->getDataItemList())
        if(name == v->getShortName()) return object(new DataItemWrapper(v));


    //throw an exception here
    throw_PyKeyError("Cannot find object ["+name+"] in group ["+
                     this->name()+"]!");

    //need this to make the compiler happy - this code will never be reached.
    return object();
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
std::string GroupWrapper::__str__() const
{
    std::stringstream ss;
    ss<<"Group ["<<this->location()<<"]";
    return ss.str();
}

//----------------------------------------------------------------------------
tuple GroupWrapper::childs() const
{
    list l;
    
    for(auto v: ptr()->getGroupList()) l.append(GroupWrapper(v));
    for(auto v: ptr()->getDataItemList()) l.append(DataItemWrapper(v));
    for(auto v: ptr()->getDimensionList()) l.append(DimensionWrapper(v));

    return tuple(l);
}

//----------------------------------------------------------------------------
TupleIterator GroupWrapper::create_iterator() 
{
    return TupleIterator(childs(),0);
}



//====================helper function to create python class==================
static const char __group_doc_parent[] = "reference to the parent group";
static const char __group_doc_root[]   = "reference to the root group";
void wrap_group()
{
    //create the wrapper for the group container class
    wrap_container<IGroupPtr>("GroupContainer");
    //create the wrapper for the attribute manager of the group 
    wrap_attribute_manager<IGroupPtr>("GroupAttributeManager");

    
    class_<GroupWrapper,bases<ContainerWrapper<IGroupPtr>> >("Group")
        .add_property("parent",&GroupWrapper::parent,__group_doc_parent)
        .add_property("root",&GroupWrapper::root,__group_doc_root)
        .def("__getitem__",&GroupWrapper::__getitem__)
        .def("__str__",&GroupWrapper::__str__)
        .def("__iter__",&GroupWrapper::create_iterator)
        ;
}


