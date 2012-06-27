#ifndef __WRAPPERHELPERS_HPP__
#define __WRAPPERHELPERS_HPP__

#include<boost/python.hpp>
#include<cdma/array/Array.h>
#include<typeinfo>

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}


using namespace cdma;
using namespace boost::python;


/*! 
\brief setup numpy module
*/
void init_numpy();

//-----------------------------------------------------------------------------
/*! 
\brief get numpy type string

Converts the type_info object to a numpy type string.
\args o instance of type_info
\return numpy type string
*/
std::string get_type_string(const std::type_info &o);

//-----------------------------------------------------------------------------
/*!
\brief get numpy type code

Return the numerical type code of a numpy type equivalent to the type
represented by o. 
\param o instance of type_info
\return numpy type code
*/
int get_type_code(const std::type_info &o);

//-----------------------------------------------------------------------------
/*! 
\brief return type size

Get the size of a type described by a type_info instance.
\param t type_info instance
\return size of type in bytes
*/
size_t get_type_size(const std::type_info &t);

//-----------------------------------------------------------------------------
/*!
\brief create a numpy array from a CDMA array

Takes a CDMA ArrayPtr and constructs a numpy array of equal shape and data type.
\param aptr pointer to a CDMA array
\return o Python object holding the numpy array
*/
object cdma2numpy_array(const ArrayPtr aptr);

//------------------------------------------------------------------------------
/*!
\brief copy data form a cdma array to a numpy array

Function copies data from a CDMA array to a numpy array. The function assumes
that the tow arrays are of same size and data type. 
\param aptr pointer to the CDMA array
\param nparray numpy array where to store the data
*/
void copy_data_from_cdma2numpy(const ArrayPtr aptr,object &nparray);

//------------------------------------------------------------------------------
/*! 
\brief throw Python TypeError exception

Throw the TypeError Python exception.
*/
void throw_PyTypeError(const std::string &message);

#endif
