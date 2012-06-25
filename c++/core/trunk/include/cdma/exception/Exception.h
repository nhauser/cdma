//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
// Contributors :
// See AUTHORS file 
//******************************************************************************
#ifndef __CDMA_EXCEPTION_H__
#define __CDMA_EXCEPTION_H__

#include <yat/Exception.h>

namespace cdma
{

//==============================================================================
/// @brief Base class for CDMA exceptions
///
/// Derivated exceptions from this base class:
///   - BadArrayTypeException
///   - DimensionNotSupportedException
///   - DivideByZeroException
///   - DuplicationException
///   - FileAccessException
///   - FitterException
///   - InvalidPointerException
///   - InvalidRangeException
///   - NoDataException
///   - NoResultException
///   - NoSignalException
///   - NotImplementedException
///   - ShapeNotMatchException
///   - TooManyResultsException
//==============================================================================
class Exception: public yat::Exception
{
public:
  Exception( const yat::Exception &yat_ex): yat::Exception(yat_ex)
  { }
  Exception( const char *reason, const char *desc, const char *origin ):
   yat::Exception(reason, desc, origin) { }
  Exception( const char *reason, const std::string& desc, const std::string& origin ):
   yat::Exception(std::string(reason), desc, origin) { }

  /// Write the errors on stdout
  void dump()
  {
    yat::Exception::dump();
  }
};

//==============================================================================
/// Macro helper to define a new exception class based on cdma::Exception
//==============================================================================
#define DECLARE_EXCEPTION(e, s) \
class e##Exception : public Exception \
{ public: \
  e##Exception(const char *desc, const char *origin ): \
   Exception(s, desc, origin) { } \
  e##Exception(const std::string& desc, const std::string& origin ): \
   Exception(s, desc, origin) { } \
}

DECLARE_EXCEPTION(DimensionNotSupported, "DIMENSION_NOT_SUPPORTED");
DECLARE_EXCEPTION(DivideByZero, "DIVIDE_BY_ZERO");
DECLARE_EXCEPTION(FileAccess, "FILE_ACCESS_ERROR");
DECLARE_EXCEPTION(Fitter, "FITTER_ERROR");
DECLARE_EXCEPTION(Duplication, "ITEM_ALREADY_EXISTS");
DECLARE_EXCEPTION(BadArrayType, "BAD_ARRAY_TYPE");
DECLARE_EXCEPTION(InvalidRange, "INVALID_RANGE");
DECLARE_EXCEPTION(ShapeNotMatch, "SHAPE_NOT_MATCH");
DECLARE_EXCEPTION(NoResult, "NO_RESULT");
DECLARE_EXCEPTION(NoData, "NO_DATA");
DECLARE_EXCEPTION(NoSignal, "NO_SIGNAL");
DECLARE_EXCEPTION(InvalidPointer, "INVALID_POINTER");
DECLARE_EXCEPTION(TooManyResults, "TOO_MANY_RESULTS");
DECLARE_EXCEPTION(NotImplemented, "NOT_IMPLEMENTED");

#define THROW_DIMENSION_NOT_SUPPORTED(m, o) \
  throw DimensionNotSupportedException(m, o)
  
#define THROW_DIVIDE_NY_ZERO(m, o) \
  throw DivideByZeroException(m, o)
  
#define THROW_FILE_ACCESS(m, o) \
  throw FileAccessException(m, o)
  
#define THROW_FITTER_ERROR(m, o) \
  throw FitterException(m, o)
  
#define THROW_DUPLICATION_ERROR(m, o) \
  throw DuplicationException(m, o)
  
#define THROW_BAD_ARRAY_TYPE(m, o) \
  throw BadArrayTypeException(m, o)
  
#define THROW_INVALID_RANGE(m, o) \
  throw InvalidRangeException(m, o)
  
#define THROW_SHAPE_NOT_MATCH(m, o) \
  throw ShapeNotMatchException(m, o)
  
#define THROW_NO_RESULT(m, o) \
  throw NoResultException(m, o)
  
#define THROW_NO_DATA(m, o) \
  throw NoDataException(m, o)
  
#define THROW_NO_SIGNAL(m, o) \
  throw NoSignalException(m, o)
  
#define THROW_INVALID_POINTER(m, o) \
  throw InvalidPointerException(m, o)
  
#define THROW_TOO_MANY_RESULTS(m, o) \
  throw TooManyResultsException(m, o)
  
#define THROW_NOT_IMPLEMENTED(o) \
  throw NotImplementedException("Operation cannot be performed", o)

} //namespace cdma
#endif //__CDMA_EXCEPTION_H__
