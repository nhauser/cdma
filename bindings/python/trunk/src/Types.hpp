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


struct TypeUtility
{  
    //obtain the typeid from the C++ type name
    static TypeID typename2typeid(const std::string &tname);
    //obtain the size of a type from typeid
    static size_t typeid2size(const TypeID &tid);
    //obtain the numpy type code from the typeid
    static int typeid2numpytc(const TypeID &tid);
    //obtain the numpy type code from the typeid
    static std::string typeid2numpystr(const TypeID &tid);
};


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
