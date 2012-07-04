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

#include<map>

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}
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


#endif
