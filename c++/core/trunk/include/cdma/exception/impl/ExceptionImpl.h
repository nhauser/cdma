//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
//
// This file is part of cdma-core library.
//
// The cdma-core library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
//
// The CDMA library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
//
// Contributors :
// See AUTHORS file 
//******************************************************************************

#ifndef __CDMA_EXCEPTION_IMPL_H__
#define __CDMA_EXCEPTION_IMPL_H__

#include "yat/Exception.h"
#include "cdma/exception/Exception.h"
#include <iostream>

namespace cdma
{

//==============================================================================
/// Base class for all cdma exception to gives the inheritance from yat::Exception
//==============================================================================
class BaseException : public yat::Exception
{
 public:
  BaseException(const yat::Exception& ex): yat::Exception(ex) {}
  BaseException(const char* reason, const char *desc, const char *origin ):
   yat::Exception(reason, desc, origin) { }
  BaseException(const char* reason, const std::string& desc, const std::string& origin ):
   yat::Exception(reason, desc, origin) { }

  void dump()
  {
    std::cout << "!!!! CDMA Exception !!!!" << std::endl;
   
    for( size_t i=0; i < errors.size(); ++i )
    {
      std::cout << "-------------------------" << std::endl;
      std::cout << "Err[" << i << "] reason.. " << errors[i].reason << std::endl;
      std::cout << "Err[" << i << "] desc.... " << errors[i].desc << std::endl;
      std::cout << "Err[" << i << "] origin.. " << errors[i].origin << std::endl;
    }
    std::cout << "-------------------------" << std::endl;
  }
};

//==============================================================================
/// General purpose exception
//==============================================================================
class GenericExceptionImpl : public GenericException, public BaseException
{ public:
  GenericExceptionImpl(const yat::Exception& ex): BaseException(ex) {}
  GenericExceptionImpl(const char *reason, const char *desc, const char *origin ):
   BaseException(reason, desc, origin) { }
  GenericExceptionImpl(const char *reason, const std::string& desc, const std::string& origin ):
   BaseException(reason, desc, origin) { }
  void dump()
  {
    BaseException::dump();
  }
};

//==============================================================================
/// Macro helper to define a new exception class based on 
/// cdma::BaseException & cdma::Exception
//==============================================================================
#define DECLARE_EXCEPTION_IMPL(e, s) \
class e##ExceptionImpl : public e##Exception, public BaseException \
{ public: \
  e##ExceptionImpl(const char *desc, const char *origin ): \
   BaseException(s, desc, origin) { } \
  e##ExceptionImpl(const std::string& desc, const std::string& origin ): \
   BaseException(s, desc, origin) { } \
  void dump() \
  { \
    BaseException::dump(); \
  } \
};

DECLARE_EXCEPTION_IMPL(DimensionNotSupported, "DIMENSION_NOT_SUPPORTED");
DECLARE_EXCEPTION_IMPL(DivideByZero, "DIVIDE_BY_ZERO");
DECLARE_EXCEPTION_IMPL(FileAccess, "FILE_ACCESS_ERROR");
DECLARE_EXCEPTION_IMPL(Fitter, "FITTER_ERROR");
DECLARE_EXCEPTION_IMPL(Duplication, "ITEM_ALREADY_EXISTS");
DECLARE_EXCEPTION_IMPL(BadArrayType, "BAD_ARRAY_TYPE");
DECLARE_EXCEPTION_IMPL(InvalidRange, "INVALID_RANGE");
DECLARE_EXCEPTION_IMPL(ShapeNotMatch, "SHAPE_NOT_MATCH");
DECLARE_EXCEPTION_IMPL(NoResult, "NO_RESULT");
DECLARE_EXCEPTION_IMPL(NoData, "NO_DATA");
DECLARE_EXCEPTION_IMPL(NoSignal, "NO_SIGNAL");
DECLARE_EXCEPTION_IMPL(InvalidPointer, "INVALID_POINTER");
DECLARE_EXCEPTION_IMPL(TooManyResults, "TOO_MANY_RESULTS");
DECLARE_EXCEPTION_IMPL(NotImplemented, "NOT_IMPLEMENTED");
DECLARE_EXCEPTION_IMPL(TypeMismatch, "TYPE_MISMATCH");

#define YAT_TO_CDMA_EXCEPTION(e) \
  throw GenericExceptionImpl(e)
  
#define RE_THROW_EXCEPTION(e) \
  throw GenericExceptionImpl(e)
  
#define THROW_EXCEPTION(r, m, o) \
  throw GenericExceptionImpl(r, m, o)
  
#define THROW_DIMENSION_NOT_SUPPORTED(m, o) \
  throw DimensionNotSupportedExceptionImpl(m, o)
  
#define THROW_DIVIDE_NY_ZERO(m, o) \
  throw DivideByZeroExceptionImpl(m, o)
  
#define THROW_FILE_ACCESS(m, o) \
  throw FileAccessExceptionImpl(m, o)
  
#define THROW_FITTER_ERROR(m, o) \
  throw FitterExceptionImpl(m, o)
  
#define THROW_DUPLICATION_ERROR(m, o) \
  throw DuplicationExceptionImpl(m, o)
  
#define THROW_BAD_ARRAY_TYPE(m, o) \
  throw BadArrayTypeExceptionImpl(m, o)
  
#define THROW_INVALID_RANGE(m, o) \
  throw InvalidRangeExceptionImpl(m, o)
  
#define THROW_SHAPE_NOT_MATCH(m, o) \
  throw ShapeNotMatchExceptionImpl(m, o)
  
#define THROW_NO_RESULT(m, o) \
  throw NoResultExceptionImpl(m, o)
  
#define THROW_NO_DATA(m, o) \
  throw NoDataExceptionImpl(m, o)
  
#define THROW_NO_SIGNAL(m, o) \
  throw NoSignalExceptionImpl(m, o)
  
#define THROW_INVALID_POINTER(m, o) \
  throw InvalidPointerExceptionImpl(m, o)
  
#define THROW_TOO_MANY_RESULTS(m, o) \
  throw TooManyResultsExceptionImpl(m, o)
  
#define THROW_TYPE_MISMATCH(m, o) \
  throw TypeMismatchExceptionImpl(m, o)

#define THROW_NOT_IMPLEMENTED(o) \
  throw NotImplementedExceptionImpl("Operation cannot be performed", o)

} //namespace cdma
#endif //__CDMA_EXCEPTION_H__
