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
#ifndef __WRAPPERHELPERS_HPP__
#define __WRAPPERHELPERS_HPP__

#include<boost/python.hpp>
#include<cdma/array/IArray.h>
#include<typeinfo>
#include<map>
#include<list>

#include "Types.hpp"
#include "Exceptions.hpp"
#include "Selection.hpp"
#include "ArrayWrapper.hpp"
#include "DimensionWrapper.hpp"


using namespace cdma;
using namespace boost::python;


/*! 
\ingroup utility_classes
\brief setup numpy module
*/
void init_numpy();


//-----------------------------------------------------------------------------
/*!
\ingroup utility_classes
\brief convert container to list

Function converting a standard C++ container to a Python list.
\param c C++ container
\return Python list
*/
template<typename CTYPE> list cont2list(const CTYPE &c)
{
    list l;

    for(typename CTYPE::const_iterator iter=c.begin();iter!=c.end();++iter)
    {
        const typename CTYPE::value_type &v = *iter;
        l.append(v);
    }

    return l;
}

//-----------------------------------------------------------------------------
/*! 
\ingroup utility_classes
\brief convert container to tuple

Function converting a standard c++ container to a Python tuple
\param c C++ container
\return Python tuple
*/
template<typename CTYPE> tuple cont2tuple(const CTYPE &c)
{
    return tuple(cont2list(c));
}

//-----------------------------------------------------------------------------
/*!
\ingroup utility_classes
\brief create a numpy array from a CDMA array

Takes a CDMA ArrayPtr and constructs a numpy array of equal shape and data type.
If the copyflag argument is true, the data from the original array will be
copied to the numpy array.
\param array reference to the array 
\param copyflag decide if data should be copied
\return o Python object holding the numpy array
*/
object cdma2numpy_array(const ArrayWrapper &array,bool copyflag=false);


//------------------------------------------------------------------------------
/*!
\ingroup utiltiy_classes
\brief converts a type to string

Template function converting a type id to a numpy type string. 
This function is intended to be used as a class method for wrapper objects that
implement the IOObject interface. 
\param self object of type implementing IOObject
\return string with numpy type string
*/
template<typename WTYPE> std::string __type__(WTYPE &o)
{
    return typeid2numpystr[o.type()];
}

//------------------------------------------------------------------------------
/*!
\ingroup utility_classes
\brief converst list to tuple as shape

Template method converting a std::vector<int> to a tuple which is used as a
shape property of an object. This function is inteded to be used as a class
method for wrappers implementing the IOobject interface.
*/
template<typename WTYPE> tuple __shape__(WTYPE &self)
{
    return cont2tuple(self.shape());
}

//------------------------------------------------------------------------------
/*! 
\ingroup utility_classes
\brief get item function

*/
template<typename WTYPE> object __getitem__(WTYPE &o,object &selection)
{
    //if the data object itself is scalar we can only return a scalar value
    //in this case we ignore all arguments to __getitem__
    if((o.shape().size()==1)||(o.type() == STRING))
            return read_scalar_data(o);

    //ok - we have a multidimensional data object. Now it depends on the 
    //selection object of what will be returned. The selection object can either
    //be a list or tuple or a single python object from which the selection must
    //be assembled. 

    Selection sel;
    if(PyTuple_Check(selection.ptr()))
        sel = create_selection(o,tuple(selection));
    else
        sel = create_selection(o,make_tuple<object>(selection));

    //std::cout<<sel<<std::endl;

    //now we have the selection we need to read data - as CDMA actually does not
    //support strides others than 1 we have to fix this here
    std::vector<size_t> offset(sel.offset());
    std::vector<size_t> shape(sel.shape());
    for(size_t i=0;i<sel.rank();i++)
        shape[i] *= sel.stride()[i];

    //read data
    ArrayWrapper array = o.get(offset,shape);

    //now we need to convert the array to a numpy array
    return cdma2numpy_array(array,true);
}

//------------------------------------------------------------------------------
/*! 
\ingroup utility_classes
\brief template reading scalar data 

This template reads scalar data from an object that provides the IOObject
interface. 
\throws TypError native python exception if data type not supported
\param o instance of WTYPE
\return python object with the scalar data
*/
template<typename WTYPE> object read_scalar_data(WTYPE &o)
{
    switch(o.type())
    {
        case BYTE: return object(o.template get<int8_t>());
        case UBYTE: return object(o.template get<uint8_t>());
        case SHORT: return object(o.template get<int16_t>());
        case USHORT: return object(o.template get<uint16_t>());
        case INT: return object(o.template get<int32_t>());
        case UINT: return object(o.template get<uint32_t>());
        case LONG: return object(o.template get<int64_t>());
        case ULONG: return object(o.template get<uint64_t>());
        case FLOAT: return object(o.template get<float>());
        case DOUBLE: return object(o.template get<double>());
        case STRING: return object(o.template get<std::string>());
        default:
            throw_PyTypeError("Data type not supported!");

    };

    return object(); //return value only to avoid compiler warnings 
                     //this code will never be reached.
}

//-----------------------------------------------------------------------------
/*! 
\ingroup utility_classes
\brief returns a list of dimensions

Converts a vector of IDimensionPtr entries to a tuple of DimensionWrapper 
objects.

*/
template<typename WTYPE> tuple __dimensions__(WTYPE &o)
{
    list l;


    std::list<IDimensionPtr> dlist = o.dimensions();
    for(std::list<IDimensionPtr>::iterator iter = dlist.begin();
             iter != dlist.end();++iter)
    {
        l.append(DimensionWrapper(*iter));
    }

    return tuple(l);
}

#endif
