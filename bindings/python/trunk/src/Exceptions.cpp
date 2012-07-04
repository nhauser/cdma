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
 * Created on: Jun 26, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

extern "C"{
#include<Python.h>
}

#include<string>
#include<boost/python.hpp>

using namespace boost::python;

#include<cdma/exception/Exception.h>
using namespace cdma;

#include "Exceptions.hpp"

#define ERR_TRANSLATOR_NAME(CDMAETYPE) CDMAETYPE ## _translator
#define ERR_PTR_NAME(CDMAETYPE) Py ## CDMAETYPE ## Ptr
#define ERR_OBJ_NAME(CDMAETYPE) Py ## CDMAETYPE

#define ERR_TRANSLATOR(CDMAETYPE)\
    PyObject *ERR_PTR_NAME(CDMAETYPE) = nullptr;\
    void ERR_TRANSLATOR_NAME(CDMAETYPE)(const CDMAETYPE &error)\
    {\
        assert(ERR_PTR_NAME(CDMAETYPE) != nullptr);\
        object exception(error);\
        PyErr_SetObject(ERR_PTR_NAME(CDMAETYPE),exception.ptr());\
    }

#define ERR_OBJECT_DECL(CDMAETYPE)\
    object ERR_OBJ_NAME(CDMAETYPE) = (\
            class_<CDMAETYPE>(# CDMAETYPE ,init<std::string,std::string>()));\
    ERR_PTR_NAME(CDMAETYPE) = ERR_OBJ_NAME(CDMAETYPE).ptr();

#define ERR_REGISTRATION(CDMAETYPE)\
    register_exception_translator<CDMAETYPE>(ERR_TRANSLATOR_NAME(CDMAETYPE)); 

ERR_TRANSLATOR(BadArrayTypeException);
ERR_TRANSLATOR(DimensionNotSupportedException);
ERR_TRANSLATOR(DivideByZeroException);
ERR_TRANSLATOR(DuplicationException);
ERR_TRANSLATOR(FileAccessException);
ERR_TRANSLATOR(FitterException);
ERR_TRANSLATOR(InvalidPointerException);
ERR_TRANSLATOR(InvalidRangeException);
ERR_TRANSLATOR(NoDataException);
ERR_TRANSLATOR(NoResultException);
ERR_TRANSLATOR(NoSignalException);
ERR_TRANSLATOR(NotImplementedException);
ERR_TRANSLATOR(ShapeNotMatchException);
ERR_TRANSLATOR(TooManyResultsException);


void exception_registration()
{
    ERR_OBJECT_DECL(BadArrayTypeException);
    ERR_OBJECT_DECL(DimensionNotSupportedException);
    ERR_OBJECT_DECL(DivideByZeroException);
    ERR_OBJECT_DECL(DuplicationException);
    ERR_OBJECT_DECL(FileAccessException);
    ERR_OBJECT_DECL(FitterException);
    ERR_OBJECT_DECL(InvalidPointerException);
    ERR_OBJECT_DECL(InvalidRangeException);
    ERR_OBJECT_DECL(NoDataException);
    ERR_OBJECT_DECL(NoResultException);
    ERR_OBJECT_DECL(NoSignalException);
    ERR_OBJECT_DECL(NotImplementedException);
    ERR_OBJECT_DECL(ShapeNotMatchException);
    ERR_OBJECT_DECL(TooManyResultsException);
    
    ERR_REGISTRATION(BadArrayTypeException);
    ERR_REGISTRATION(DimensionNotSupportedException);
    ERR_REGISTRATION(DivideByZeroException);
    ERR_REGISTRATION(DuplicationException);
    ERR_REGISTRATION(FileAccessException);
    ERR_REGISTRATION(FitterException);
    ERR_REGISTRATION(InvalidPointerException);
    ERR_REGISTRATION(InvalidRangeException);
    ERR_REGISTRATION(NoDataException);
    ERR_REGISTRATION(NoResultException);
    ERR_REGISTRATION(NoSignalException);
    ERR_REGISTRATION(NotImplementedException);
    ERR_REGISTRATION(ShapeNotMatchException);
    ERR_REGISTRATION(TooManyResultsException);

}

//-----------------------------------------------------------------------------
void throw_PyTypeError(const std::string &message)
{
    PyErr_SetString(PyExc_TypeError,message.c_str());
    throw error_already_set();
}

//-----------------------------------------------------------------------------
void throw_PyIndexError(const std::string &message)
{
    PyErr_SetString(PyExc_IndexError,message.c_str());
    throw error_already_set();
}

//-----------------------------------------------------------------------------
void throw_PyKeyError(const std::string &message)
{
    PyErr_SetString(PyExc_KeyError,message.c_str());
    throw error_already_set();
}


