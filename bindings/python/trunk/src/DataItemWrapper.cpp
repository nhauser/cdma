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

#include "DataItemWrapper.hpp"
#include "AttributeManager.hpp"

//======================wrapper methods implementation=========================
std::vector<size_t> DataItemWrapper::shape() const
{
    std::vector<size_t> shape;

    for(auto v: ptr()->getShape())
    {
        shape.push_back(v);
    }

    return shape;
}

//-----------------------------------------------------------------------------
TypeID DataItemWrapper::type() const
{
    return typename2typeid[ptr()->getType().name()];
}

//================overloaded scalar get template===============================
template<> std::string DataItemWrapper::get<std::string>() const
{
    std::cout<<"Calling get string method ..."<<std::endl;
    return ptr()->readString();
}

//-----------------------------------------------------------------------------
std::list<IDimensionPtr> DataItemWrapper::dimensions() const
{
    return ptr()->getDimensionList();
}

//-----------------------------------------------------------------------------
std::string DataItemWrapper::__str__() const
{
    std::stringstream ss;

    ss<<"DataItem ["<<this->name()<<"] type="<<typeid2numpystr[this->type()];
    ss<<" shape=( ";
    for(auto v: this->shape()) ss<<v<<" ";
    ss<<")";
    return ss.str();
}

//-----------------------------------------------------------------------------
ArrayWrapper DataItemWrapper::get(const std::vector<size_t> &offset,
                                  const std::vector<size_t> & shape)
{
    std::vector<int> _offset(offset.size());
    std::vector<int> _shape(shape.size());
    for(size_t i=0;i<offset.size();i++)
    {
        _offset[i] = offset[i];
        _shape[i]  = shape[i];
    }
    for(auto v: _offset) std::cout<<v<<" ";
    std::cout<<std::endl;
    for(auto v: _shape) std::cout<<v<<" ";
    std::cout<<std::endl;

    return ArrayWrapper(ptr()->getData(_offset,_shape)->deepCopy());
}


//===============helper function creating the python class=====================
static const char __dataitem_doc_dim [] = 
"Dimension manager attribute of the data item";
static const char __dataitem_doc_rank [] = 
"number of dimensions (rank) of the dataitem";
static const char __dataitem_doc_shape [] =
"number of elements along each dimension (shape) of the data item";
static const char __dataitem_doc_size [] = 
"total number of elements in the data item";
static const char __dataitem_doc_unit [] = 
"string representing the physical unit of the data stored in the item";
static const char __dataitem_doc_type [] = 
"string with the numpy type used to store the data";
static const char __dataitem_doc_description [] = 
"description of the dataitem";
void wrap_dataitem()
{
    wrap_container<IDataItemPtr>("DataItemContainer");
    wrap_attribute_manager<IDataItemPtr>("DataItemAttributeManager");
    wrap_dimensionmanager();

    class_<DataItemWrapper,bases<ContainerWrapper<IDataItemPtr>> >("DataItem")
        .def_readwrite("dim",&DataItemWrapper::dim,__dataitem_doc_dim)
        .add_property("rank",&DataItemWrapper::rank,__dataitem_doc_rank)
        .add_property("shape",&__shape__<DataItemWrapper>,__dataitem_doc_shape)
        .add_property("size",&DataItemWrapper::size,__dataitem_doc_size)
        .add_property("unit",&DataItemWrapper::unit,__dataitem_doc_unit)
        .add_property("type",&__type__<DataItemWrapper>,__dataitem_doc_type)
        .add_property("description",&DataItemWrapper::description,
                      __dataitem_doc_description)
        .add_property("dims",&__dimensions__<DataItemWrapper>)
        .def("__getitem__",&__getitem__<DataItemWrapper>)
        .def("__str__",&DataItemWrapper::__str__)
        ;
        
}
