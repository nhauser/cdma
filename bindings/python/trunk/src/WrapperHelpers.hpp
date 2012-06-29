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
        {typeid(double).name(),TypeID::DOUBLE},
        {typeid(std::string).name(),TypeID::STRING}};

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



/*! 
\brief setup numpy module
*/
void init_numpy();

//specialization for ArrayPtr
std::string get_type_string(ArrayPtr ptr);

//specialization for ArrayPtr
int get_type_code(ArrayPtr ptr);



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
object cdma2numpy_array(const ArrayPtr aptr,bool copyflag=false);


//------------------------------------------------------------------------------
/*! 
\brief throw Python TypeError exception

Throw the TypeError Python exception.
*/
void throw_PyTypeError(const std::string &message);

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
    if(o.shape().size()==0) return read_scalar_data(o);

    //ok - we have a multidimensional data object. Now it depends on the 
    //selection object of what will be returned. The selection object can either
    //be a list or tuple or a single python object from which the selection must
    //be assembled. 
    //
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
