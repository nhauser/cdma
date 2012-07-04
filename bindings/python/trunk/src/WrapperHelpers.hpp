#ifndef __WRAPPERHELPERS_HPP__
#define __WRAPPERHELPERS_HPP__

#include<boost/python.hpp>
#include<cdma/array/Array.h>
#include<typeinfo>
#include<map>

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include "Types.hpp"
#include "Exceptions.hpp"
#include "Selection.hpp"
#include "ArrayWrapper.hpp"


using namespace cdma;
using namespace boost::python;





/*! 
\brief setup numpy module
*/
void init_numpy();




//-----------------------------------------------------------------------------
/*! 
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
\brief convert container to list

Function converting a standard C++ container to a Python list.
\param c C++ container
\return Python list
*/
template<typename CTYPE> list cont2list(const CTYPE &c)
{
    list l;

#ifdef NOFOREACH
    for(auto iter=c.begin();iter!=c.end();iter++)
    {
        const typename CTYPE::value_type &v = *iter;
#else
    for(auto v: c)
    {
#endif
        l.append(v);
    }

    return l;
}

//-----------------------------------------------------------------------------
/*!
\brief create a numpy array from a CDMA array

Takes a CDMA ArrayPtr and constructs a numpy array of equal shape and data type.
\param aptr pointer to a CDMA array
\return o Python object holding the numpy array
*/
object cdma2numpy_array(const ArrayWrapper &array,bool copyflag=false);


//------------------------------------------------------------------------------
/*!
\brief converts a type to string

Template function converting a type id to a numpy type string. 
This function is intended to be used as a class method for wrapper objects that
implement the IOObject interface. 
\param self object of type implementing IOObject
\return string with numpy type string
*/
template<typename WTYPE> std::string __type__(WTYPE &self)
{
    return typeid2numpystr[self.type()];
}

//------------------------------------------------------------------------------
/*!
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
template<typename WTYPE> object __getitem__(WTYPE &o,object &selection)
{
    //if the data object itself is scalar we can only return a scalar value
    //in this case we ignore all arguments to __getitem__
    if(o.shape().size()==0) return read_scalar_data(o);

    //ok - we have a multidimensional data object. Now it depends on the 
    //selection object of what will be returned. The selection object can either
    //be a list or tuple or a single python object from which the selection must
    //be assembled. 

    Selection sel;
    if(PyTuple_Check(selection.ptr()))
        sel = create_selection(o,tuple(selection));
    else
        sel = create_selection(o,make_tuple<object>(selection));

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
        case TypeID::BYTE: return object(o.template get<int8_t>());
        case TypeID::UBYTE: return object(o.template get<uint8_t>());
        case TypeID::SHORT: return object(o.template get<int16_t>());
        case TypeID::USHORT: return object(o.template get<uint16_t>());
        case TypeID::INT: return object(o.template get<int32_t>());
        case TypeID::UINT: return object(o.template get<uint32_t>());
        case TypeID::LONG: return object(o.template get<int64_t>());
        case TypeID::ULONG: return object(o.template get<uint64_t>());
        case TypeID::FLOAT: return object(o.template get<float>());
        case TypeID::DOUBLE: return object(o.template get<double>());
        case TypeID::STRING: return object(o.template get<std::string>());
        default:
            throw_PyTypeError("Data type not supported!");

    };

    return object(); //return value only to avoid compiler warnings 
                     //this code will never be reached.
}

#endif
