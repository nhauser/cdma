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

//! enum class with type IDs.
enum class TypeID { BYTE,   //!< signded char (8Bit)
                    UBYTE,  //!< unsigned char (8Bit)
                    SHORT,  //!< signed short (16Bit)
                    USHORT, //!< unsigned short (16Bit)
                    INT,    //!< integer (32Bit)
                    UINT,   //!< unsigned integer (32Bit)
                    LONG,   //!< long (64Bit)
                    ULONG,  //!< unsigned long (64Bit)
                    FLOAT,  //!< IEEE floating point (32Bit)
                    DOUBLE, //!< IEEE floating point (64Bit)
                    STRING  //!< String type
                  };

//conversion map from names of numerical native types to TypeID
static std::map<std::string,TypeID> typename2typeid = {
        {typeid(int8_t).name(),TypeID::BYTE},
        {typeid(uint8_t).name(),TypeID::UBYTE},
        {typeid(int16_t).name(),TypeID::SHORT},
        {typeid(uint16_t).name(),TypeID::USHORT},
        {typeid(int32_t).name(),TypeID::INT},
        {typeid(uint32_t).name(),TypeID::UINT},
        {typeid(float).name(),TypeID::FLOAT},
        {typeid(double).name(),TypeID::DOUBLE}};

//conversion map from Type IDs to type sizes
static std::map<TypeID,size_t> typeid2size = {
        {TypeID::BYTE,sizeof(int8_t)}, {TypeID::UBYTE,sizeof(uint8_t)},
        {TypeID::SHORT,sizeof(int16_t)}, {TypeID::USHORT,sizeof(uint16_t)},
        {TypeID::INT,sizeof(int32_t)}, {TypeID::UINT,sizeof(uint32_t)},
        {TypeID::FLOAT,sizeof(float)}, {TypeID::DOUBLE,sizeof(double)},
        {TypeID::STRING,sizeof(char)}};

//conversion map from Type IDs to numpy type codes
static std::map<TypeID,int> typeid2numpytc = {
        {TypeID::BYTE,NPY_BYTE}, {TypeID::UBYTE,NPY_UBYTE},
        {TypeID::SHORT,NPY_SHORT}, {TypeID::USHORT,NPY_USHORT},
        {TypeID::INT,NPY_INT}, {TypeID::UINT,NPY_UINT},
        {TypeID::LONG,NPY_LONG}, {TypeID::ULONG,NPY_ULONG},
        {TypeID::FLOAT,NPY_FLOAT}, {TypeID::DOUBLE,NPY_DOUBLE}};

//conversion map from Type IDs to numpy type strings
static std::map<TypeID,std::string> typeid2numpystr = {
        {TypeID::BYTE,"int8"}, {TypeID::UBYTE,"uint8"},
        {TypeID::SHORT,"int16"}, {TypeID::USHORT,"uint16"},
        {TypeID::INT,"int32"}, {TypeID::UINT,"uint32"},
        {TypeID::LONG,"int64"}, {TypeID::ULONG,"uint64"},
        {TypeID::FLOAT,"float32"}, {TypeID::DOUBLE,"float64"},
        {TypeID::STRING,"string"}};


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

//map from numpy type codes to type sizes
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

//specialization for ArrayPtr
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

//specialization for ArrayPtr
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

//specialization for ArrayPtr
size_t get_type_size(ArrayPtr ptr);

//-----------------------------------------------------------------------------
/*! 
\brief convert container to tuple

Function converting a standard c++ container to a Python tuple
\param c C++ container
\return Python tuple
*/
template<typename CTYPE> tuple cont2tuple(const CTYPE &c)
{
    list l(0);

    for(auto v: c) l.append(v);
    return tuple(l);
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
    list l(0);

    for(auto v: c) l.append(v);
    return l;
}

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

//------------------------------------------------------------------------------
template<typename WTYPE> std::string __type__(WTYPE &self)
{
    return typeid2numpystr[self.type()];
}

//------------------------------------------------------------------------------
template<typename WTYPE> object __getitem__(WTYPE &o,object &selection)
{
    return read_scalar_data(o);
}

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
    };
}

#endif
