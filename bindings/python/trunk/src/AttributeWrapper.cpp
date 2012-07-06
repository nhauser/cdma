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
    if(_ptr->isString()) return TypeID::STRING;
    
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
template<> float AttributeWrapper::get<float>() const
{
    return _ptr->getFloatValue();
}

//------------------------------------------------------------------------------
template<> int AttributeWrapper::get<int>() const
{
    return _ptr->getIntValue();
}

//------------------------------------------------------------------------------
template<> std::string AttributeWrapper::get<std::string>() const
{
    return _ptr->getStringValue();
}

//-----------------------------------------------------------------------------
std::string AttributeWrapper::__str__() const
{
    std::stringstream ss;
    ss<<"Attribute ["<<name()<<"] type="<<typeid2numpystr[type()];
    ss<<" shape=( ";
    for(auto v: shape()) ss<<v<<" ";
    ss<<")";
    return ss.str();
}

//========================wrap attribute objects===============================
void wrap_attribute()
{
    class_<AttributeWrapper>("Attribute")
        .add_property("size",&AttributeWrapper::size)
        .add_property("name",&AttributeWrapper::name)
        .add_property("shape",&__shape__<AttributeWrapper>)
        .add_property("type",&__type__<AttributeWrapper>)
        .def("__getitem__",&__getitem__<AttributeWrapper>)
        .def("__str__",&AttributeWrapper::__str__)
        ;
}
