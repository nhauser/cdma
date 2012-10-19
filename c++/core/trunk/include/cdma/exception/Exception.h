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

#ifndef __CDMA_EXCEPTION_H__
#define __CDMA_EXCEPTION_H__

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
class Exception
{
public:
  /// Write the errors on stdout
  virtual void dump()=0;
};

#define DECLARE_EXCEPTION(e) \
class e##Exception : public Exception {};

DECLARE_EXCEPTION(DimensionNotSupported);
DECLARE_EXCEPTION(DivideByZero);
DECLARE_EXCEPTION(FileAccess);
DECLARE_EXCEPTION(Fitter);
DECLARE_EXCEPTION(Duplication);
DECLARE_EXCEPTION(BadArrayType);
DECLARE_EXCEPTION(InvalidRange);
DECLARE_EXCEPTION(ShapeNotMatch);
DECLARE_EXCEPTION(NoResult);
DECLARE_EXCEPTION(NoData);
DECLARE_EXCEPTION(NoSignal);
DECLARE_EXCEPTION(InvalidPointer);
DECLARE_EXCEPTION(TooManyResults);
DECLARE_EXCEPTION(NotImplemented);
DECLARE_EXCEPTION(TypeMismatch);
DECLARE_EXCEPTION(Generic);

} //namespace cdma
#endif //__CDMA_EXCEPTION_H__
