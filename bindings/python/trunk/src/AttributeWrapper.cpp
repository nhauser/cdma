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
 * Created on: Jun 27, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#include <sstream>
#include "AttributeWrapper.hpp"

//-----------------------------------------------------------------------------
TypeID AttributeWrapper::type() const
{
    //if(_ptr->isString()) return TypeID::STRING;
    
    return typename2typeid[_ptr->getType().name()];
}

//-----------------------------------------------------------------------------
std::vector<size_t> AttributeWrapper::shape() const
{
    std::vector<size_t> shape;

    return shape;
}

//------------------------------------------------------------------------------
size_t AttributeWrapper::rank() const
{
    return shape().size();
}

//------------------------------------------------------------------------------
/*
template<> float AttributeWrapper::get<float>() const
{
    return _ptr->getFloatValue();
}
*/

//------------------------------------------------------------------------------
/*
template<> int AttributeWrapper::get<int>() const
{
    return _ptr->getIntValue();
}
*/

//------------------------------------------------------------------------------
/*
template<> std::string AttributeWrapper::get<std::string>() const
{
    return _ptr->getStringValue();
}
*/

//-----------------------------------------------------------------------------
std::string AttributeWrapper::__str__() const
{
    std::stringstream ss;
    ss<<"Attribute ["<<name()<<"] type="<<typeid2numpystr[type()];
    ss<<" shape=( ";
    std::vector<size_t> s = shape();
    for(std::vector<size_t>::iterator iter = s.begin(); iter!=s.end();++iter)
        ss<<*iter<<" ";

    ss<<")";
    return ss.str();
}

//========================wrap attribute objects===============================
static const char __attribute_doc_class [] = 
"The attribute class represents a CDMA attribute";
static const char __attribute_doc_size [] = 
"the size of the attribute (number of elements)";
static const char __attribute_doc_name [] = 
"the attributes name";
static const char __attribute_doc_shape [] = 
"shape of the attribute as tuple";
static const char __attribute_doc_type [] = 
"the data type of the attribute as numpy type code";

void wrap_attribute()
{
    class_<AttributeWrapper>("Attribute",__attribute_doc_class)
        .add_property("size",&AttributeWrapper::size,__attribute_doc_size)
        .add_property("name",&AttributeWrapper::name,__attribute_doc_name)
        .add_property("shape",&__shape__<AttributeWrapper>,__attribute_doc_shape)
        .add_property("type",&__type__<AttributeWrapper>,__attribute_doc_type)
        .def("__getitem__",&__getitem__<AttributeWrapper>)
        .def("__str__",&AttributeWrapper::__str__)
        ;
}
