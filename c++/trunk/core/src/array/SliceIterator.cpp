// ******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
// ******************************************************************************

#include <cdma/array/impl/SliceIterator.h>
#include <cdma/array/IArray.h>

namespace cdma
{

//---------------------------------------------------------------------------
// SliceIterator::hasNext
//---------------------------------------------------------------------------
bool SliceIterator::hasNext() { THROW_NOT_IMPLEMENTED("SliceIterator::hasNext"); }

//---------------------------------------------------------------------------
// SliceIterator::next
//---------------------------------------------------------------------------
void SliceIterator::next() { THROW_NOT_IMPLEMENTED("SliceIterator::next"); }

//---------------------------------------------------------------------------
// SliceIterator::getArrayNext
//---------------------------------------------------------------------------
IArrayPtr SliceIterator::getArrayNext() throw ( cdma::Exception ) { THROW_NOT_IMPLEMENTED("SliceIterator::getArrayNext"); }

//---------------------------------------------------------------------------
// SliceIterator::getArrayCurrent
//---------------------------------------------------------------------------
IArrayPtr SliceIterator::getArrayCurrent() throw ( cdma::Exception ) { THROW_NOT_IMPLEMENTED("SliceIterator::getArrayCurrent"); }

//---------------------------------------------------------------------------
// SliceIterator::getSliceShape
//---------------------------------------------------------------------------
std::vector<int> SliceIterator::getSliceShape() throw ( cdma::Exception ) { THROW_NOT_IMPLEMENTED("SliceIterator::getSliceShape"); }

//---------------------------------------------------------------------------
// SliceIterator::getSlicePosition
//---------------------------------------------------------------------------
std::vector<int> SliceIterator::getSlicePosition() { THROW_NOT_IMPLEMENTED("SliceIterator::getSlicePosition"); }

}