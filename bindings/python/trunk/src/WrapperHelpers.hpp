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


using namespace cdma;
using namespace boost::python;

//map from compiler type names to numpy type codes
static std::map<std::string,int> typename2numpytc = 
         {{typeid(int8_t).name(),NPY_BYTE},
         {typeid(uint8_t).name(),NPY_UBYTE},
         {typeid(int16_t).name(),NPY_SHORT},
         {typeid(uint16_t).name(),NPY_USHORT},
         {typeid(int32_t).name(),NPY_INT},
         {typeid(uint32_t).name(),NPY_UINT},
         {typeid(int64_t).name(),NPY_LONG},
         {typeid(uint64_t).name(),NPY_ULONG},
         {typeid(float).name(),NPY_FLOAT},
         {typeid(double).name(),NPY_DOUBLE}};

//map from numpy type codes to numpy type strings
static std::map<int,std::string> numpytc2numpystr = 
         {{NPY_BYTE,"int8"},{NPY_UBYTE,"uint8"},
         {NPY_SHORT,"int16"},
         {NPY_USHORT,"uint16"},
         {NPY_INT,"int32"},{NPY_UINT,"uint32"},
         {NPY_LONG,"int64"},
         {NPY_ULONG,"uint64"},
         {NPY_FLOAT,"float32"},
         {NPY_DOUBLE,"float64"}};

static std::map<int,size_t> numpytc2size = 
        {{NPY_BYTE,sizeof(int8_t)},
        {NPY_UBYTE,sizeof(uint8_t)},
        {NPY_SHORT,sizeof(int16_t)},
        {NPY_USHORT,sizeof(uint16_t)},
        {NPY_INT,sizeof(int32_t)},
        {NPY_UINT,sizeof(uint32_t)},
        {NPY_LONG,sizeof(int64_t)},
        {NPY_ULONG,sizeof(uint64_t)},
        {NPY_FLOAT,sizeof(float)},
        {NPY_DOUBLE,sizeof(double)}};


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
template<typename CDMAPTRT> std::string get_type_string(CDMAPTRT ptr)
{
    return numpytc2numpystr[typename2numpytc[ptr->getType().name()]];
}

std::string get_type_string(ArrayPtr ptr);

//-----------------------------------------------------------------------------
/*!
\brief get numpy type code

Return the numerical type code of a numpy type equivalent to the type
represented by o. 
\param o instance of type_info
\return numpy type code
*/
template<typename CDMAPTRT> int get_type_code(CDMAPTRT ptr)
{
    return typename2numpytc[ptr->getType().name()];
}

int get_type_code(ArrayPtr ptr);

//-----------------------------------------------------------------------------
/*! 
\brief return type size

Get the size of a type described by a type_info instance.
\param t type_info instance
\return size of type in bytes
*/

template<typename CDMAPTRT> size_t get_type_size(CDMAPTRT ptr)
{
    return numpytc2size[typename2numpytc[ptr->getType().name()]];
}

size_t get_type_size(ArrayPtr ptr);
//-----------------------------------------------------------------------------
/*!
\brief create a numpy array from a CDMA array

Takes a CDMA ArrayPtr and constructs a numpy array of equal shape and data type.
\param aptr pointer to a CDMA array
\return o Python object holding the numpy array
*/
object cdma2numpy_array(const ArrayPtr aptr,bool copyflag=false);

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
