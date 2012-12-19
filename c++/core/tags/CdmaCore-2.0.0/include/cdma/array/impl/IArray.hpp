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

/// @cond excluded_from_doxygen_generation

#ifndef __CDMA_IARRAY_HPP__
#define __CDMA_IARRAY_HPP__

#include <string.h>
#include <sstream>

#include <cdma/array/ArrayIterator.h>
#include <cdma/array/IView.h>
#include <cdma/array/impl/Array.h>

namespace cdma
{
// Redirection to the concrete implementation,
// This kind of static polymorphism is possible because the unique derivation of the IArray
// interface is the Array class
#define THIS_IMPL (static_cast<Array*>(this))

//----------------------------------------------------------------------------
// Array::setValue
//----------------------------------------------------------------------------
template<typename T> void IArray::setValue(const ArrayIterator& target, T value)
{
  THIS_IMPL->setValue<T>( target, value );
}

//----------------------------------------------------------------------------
// Array::setValue
//----------------------------------------------------------------------------
template<typename T> void IArray::setValue(const IViewPtr& view, std::vector<int> position, T value)
{
  THIS_IMPL->setValue<T>( view, position, value );
}

//----------------------------------------------------------------------------
// Array::setValue
//----------------------------------------------------------------------------
template<typename T> void IArray::setValue(std::vector<int> position, T value)
{
  THIS_IMPL->setValue<T>( position, value );
}

//----------------------------------------------------------------------------
// Array::setValue
//----------------------------------------------------------------------------
template<typename T> void IArray::setValue(T value)
{
  THIS_IMPL->setValue<T>( value );
}

//----------------------------------------------------------------------------
// Array::getValue
//----------------------------------------------------------------------------
template<typename T> T IArray::getValue( std::vector<int> position )
{
  return THIS_IMPL->getValue<T>(position );
}

//----------------------------------------------------------------------------
// Array::getValue
//----------------------------------------------------------------------------
template<typename T> T IArray::getValue( void )
{
  return THIS_IMPL->getValue<T>();
}

//----------------------------------------------------------------------------
// Array::getValue<T>
//----------------------------------------------------------------------------
template<typename T> T IArray::getValue( const IViewPtr& view, std::vector<int> position )
{
  return THIS_IMPL->getValue<T>(view, position);
}
}

#endif // __CDMA_IARRAY_HPP__

/// @endcond
