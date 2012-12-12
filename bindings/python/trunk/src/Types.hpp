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
void init_typename2typeid();

//conversion map from Type IDs to type sizes
static std::map<TypeID,size_t> typeid2size;

void init_typeid2size();

//conversion map from Type IDs to numpy type codes
static std::map<TypeID,int> typeid2numpytc;

void init_typeid2numpytc();

//conversion map from Type IDs to numpy type strings
static std::map<TypeID,std::string> typeid2numpystr;

void init_typeid2numpystr();

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
