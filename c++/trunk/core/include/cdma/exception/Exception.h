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
/// Exception
/// Base class for exceptions, just a yat exception !
//==============================================================================
typedef yat::Exception Exception;

#define THROW_DIMENSION_NOT_SUPPORTED(m, o) \
  throw cdma::Exception("DIMENSION_NOT_SUPPORTED", m, o)
  
#define THROW_DIVIDE_NY_ZERO(m, o) \
  throw cdma::Exception("DIVIDE_BY_ZERO_ERROR", m, o)
  
#define THROW_FILE_ACCESS(m, o) \
  throw cdma::Exception("FILE_ACCESS_ERROR", m, o)
  
#define THROW_FITTER_ERROR(m, o) \
  throw cdma::Exception("FITTER_ERROR", m, o)
  
#define THROW_DUPLICATION_ERROR(m, o) \
  throw cdma::Exception("ITEM_ALREADY_EXISTS", m, o)
  
#define THROW_BAD_ARRAY_TYPE(m, o) \
  throw cdma::Exception("BAD_ARRAY_TYPE", m, o)
  
#define THROW_INVALID_RANGE(m, o) \
  throw cdma::Exception("INVALID_RANGE", m, o)
  
#define THROW_SHAPE_NOT_MATCH(m, o) \
  throw cdma::Exception("SHAPE_NOT_MATCH", m, o)
  
#define THROW_NO_RESULT(m, o) \
  throw cdma::Exception("NO_RESULT", m, o)
  
#define THROW_NO_DATA(m, o) \
  throw cdma::Exception("NO_DATA", m, o)
  
#define THROW_NO_SIGNAL(m, o) \
  throw cdma::Exception("NO_SIGNAL", m, o)
  
#define THROW_TOO_MANY_RESULTS(m, o) \
  throw cdma::Exception("TOO_MANY_RESULTS", m, o)
  
#define THROW_NOT_IMPLEMENTED(o) \
  throw cdma::Exception("NOT_IMPLEMENTED", "Operation cannot be performed", o)

} //namespace cdma
#endif //__CDMA_EXCEPTION_H__
