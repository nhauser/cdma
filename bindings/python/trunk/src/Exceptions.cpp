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


#include "Exceptions.hpp"

#define CDMA_EXCEPTION_NAME(EXNAME) EXNAME ## Impl
#define ERR_TRANSLATOR_NAME(CDMAETYPE) CDMAETYPE ## _translator
#define ERR_PTR_NAME(CDMAETYPE) Py ## CDMAETYPE ## Ptr
#define ERR_OBJ_NAME(CDMAETYPE) Py ## CDMAETYPE

#define ERR_TRANSLATOR(CDMAETYPE)\
    PyObject *ERR_PTR_NAME(CDMAETYPE) = NULL;\
    void ERR_TRANSLATOR_NAME(CDMAETYPE)(const CDMAETYPE &error)\
    {\
        assert(ERR_PTR_NAME(CDMAETYPE) != NULL);\
        object exception(error);\
        PyErr_SetObject(ERR_PTR_NAME(CDMAETYPE),exception.ptr());\
    }

#define ERR_OBJECT_DECL(CDMAETYPE)\
    object ERR_OBJ_NAME(CDMAETYPE) = (\
            class_<CDMAETYPE>(# CDMAETYPE ,init<std::string,std::string>()));\
    ERR_PTR_NAME(CDMAETYPE) = ERR_OBJ_NAME(CDMAETYPE).ptr();

#define ERR_REGISTRATION(CDMAETYPE)\
    register_exception_translator<CDMAETYPE>(ERR_TRANSLATOR_NAME(CDMAETYPE)); 

ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(BadArrayTypeException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(DimensionNotSupportedException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(DivideByZeroException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(DuplicationException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(FileAccessException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(FitterException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(InvalidPointerException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(InvalidRangeException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(NoDataException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(NoResultException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(NoSignalException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(NotImplementedException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(ShapeNotMatchException));
ERR_TRANSLATOR(CDMA_EXCEPTION_NAME(TooManyResultsException));


void exception_registration()
{
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(BadArrayTypeException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(DimensionNotSupportedException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(DivideByZeroException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(DuplicationException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(FileAccessException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(FitterException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(InvalidPointerException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(InvalidRangeException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(NoDataException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(NoResultException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(NoSignalException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(NotImplementedException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(ShapeNotMatchException));
    ERR_OBJECT_DECL(CDMA_EXCEPTION_NAME(TooManyResultsException));
    
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(BadArrayTypeException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(DimensionNotSupportedException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(DivideByZeroException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(DuplicationException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(FileAccessException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(FitterException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(InvalidPointerException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(InvalidRangeException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(NoDataException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(NoResultException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(NoSignalException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(NotImplementedException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(ShapeNotMatchException));
    ERR_REGISTRATION(CDMA_EXCEPTION_NAME(TooManyResultsException));

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

//-----------------------------------------------------------------------------
void throw_PyStopIteration(const std::string &message)
{
    PyErr_SetString(PyExc_StopIteration,"stop iteration");
    throw error_already_set();
}

