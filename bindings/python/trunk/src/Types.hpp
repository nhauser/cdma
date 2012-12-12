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
 * Created on: Jul 03, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include<iostream>
#include<initializer_list>
#include<typeinfo>
#include<map>

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}
/*! 
\ingroup utility_classes
\brief enum class with type IDs

This enum class provides CDMA type IDs.
*/
enum  TypeID { BYTE,   //!< signded char (8Bit)
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

//! output operator
std::ostream &operator<<(std::ostream &o,const TypeID &tid);


//conversion map from names of numerical native types to TypeID
static std::map<std::string,TypeID> typename2typeid;
void init_typename2typeid()
{
    typename2typeid[typeid(int8_t).name()  ]  = BYTE;
    typename2typeid[typeid(uint8_t).name() ]  = UBYTE;
    typename2typeid[typeid(int16_t).name() ]  = SHORT;
    typename2typeid[typeid(uint16_t).name()]  = USHORT;
    typename2typeid[typeid(int32_t).name() ]  = INT;
    typename2typeid[typeid(uint32_t).name()]  = UINT;
    typename2typeid[typeid(int64_t).name() ]  = LONG;
    typename2typeid[typeid(uint64_t).name()]  = ULONG;
    typename2typeid[typeid(float).name()   ]  = FLOAT;
    typename2typeid[typeid(double).name()  ]  = DOUBLE;
    typename2typeid[typeid(std::string).name() ]  = STRING;
}

//conversion map from Type IDs to type sizes
static std::map<TypeID,size_t> typeid2size;

void init_typeid2size()
{
    typeid2size[BYTE] = sizeof(int8_t);
    typeid2size[UBYTE] =  sizeof(uint8_t);
    typeid2size[SHORT] = sizeof(int16_t);
    typeid2size[USHORT] = sizeof(uint16_t);
    typeid2size[INT]  = sizeof(int32_t);
    typeid2size[UINT]  = sizeof(uint32_t);
    typeid2size[LONG]  = sizeof(int64_t);
    typeid2size[ULONG] = sizeof(uint64_t);
    typeid2size[FLOAT] = sizeof(float);
    typeid2size[DOUBLE] = sizeof(double);
    typeid2size[STRING] = sizeof(std::string);
}

//conversion map from Type IDs to numpy type codes
static std::map<TypeID,int> typeid2numpytc;

void init_typeid2numpytc()
{
    typeid2numpytc[BYTE] = NPY_BYTE;
    typeid2numpytc[UBYTE] = NPY_UBYTE;
    typeid2numpytc[SHORT] = NPY_SHORT;
    typeid2numpytc[USHORT] = NPY_USHORT;
    typeid2numpytc[INT] = NPY_INT;
    typeid2numpytc[UINT] = NPY_UINT;
    typeid2numpytc[LONG]  = NPY_LONG;
    typeid2numpytc[ULONG] = NPY_ULONG;
    typeid2numpytc[FLOAT] = NPY_FLOAT;
    typeid2numpytc[DOUBLE] = NPY_DOUBLE;
}

//conversion map from Type IDs to numpy type strings
static std::map<TypeID,std::string> typeid2numpystr;

void init_typeid2numpystr()
{
    typeid2numpystr[BYTE] = "int8";
    typeid2numpystr[UBYTE] = "uint8";
    typeid2numpystr[SHORT] = "int16";
    typeid2numpystr[USHORT] = "uint16";
    typeid2numpystr[INT] = "int32";
    typeid2numpystr[UINT] = "uint32";
    typeid2numpystr[LONG] = "int64";
    typeid2numpystr[ULONG] = "uint64";
    typeid2numpystr[FLOAT] = "float32";
    typeid2numpystr[DOUBLE]  = "float64";
    typeid2numpystr[STRING] = "string";
}

/*!
\ingroup type_classes
\brief < operator for TypeID

gcc 4.4 does not implement the < operator for scoped enums. In such cases
this overloaded version is used. This operator will only be used if the code
is compiled with \c -DENUMBUG.
*/
bool operator<(TypeID a,TypeID b);
bool operator>(TypeID a,TypeID b);
bool operator<=(TypeID a,TypeID b);
bool operator>=(TypeID a,TypeID b);


#endif
