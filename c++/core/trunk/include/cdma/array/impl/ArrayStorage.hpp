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

#ifndef __CDMA_DEFAULTARRAYSTORAGE_HPP__
#define __CDMA_DEFAULTARRAYSTORAGE_HPP__

#include <string.h>
#include <cstdlib>

/// @cond excluded from documentation

namespace cdma
{

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::~DefaultArrayStorage
//----------------------------------------------------------------------------
template<typename T> DefaultArrayStorage<T>::~DefaultArrayStorage()
{
    CDMA_FUNCTION_TRACE("DefaultArrayStorage::~DefaultArrayStorage");
    delete [] m_data;
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::DefaultArrayStorage
//----------------------------------------------------------------------------
template<typename T> DefaultArrayStorage<T>::DefaultArrayStorage( T* data,
        size_t length )
{
  m_data = data;
  m_elem_size = sizeof(T);
  m_array_length = length;
  m_dirty = false;
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::DefaultArrayStorage
//----------------------------------------------------------------------------
template<typename T> DefaultArrayStorage<T>::DefaultArrayStorage( T* data, std::vector<int> shape )
{
  m_data = data;
  m_elem_size = sizeof(T);
  m_array_length = 1;
  for( yat::uint16 i = 0; i < shape.size(); i++ )
  {
    m_array_length *= shape[i];
  }
  m_dirty = false;
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::get
//----------------------------------------------------------------------------
template<typename T> void* DefaultArrayStorage<T>::getValue( const cdma::IViewPtr& view, std::vector<int> position )
{
  long idx = view->getElementOffset(position);
  return (void*)(&m_data[idx]);
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::set
//----------------------------------------------------------------------------
template<typename T> void DefaultArrayStorage<T>::setValue(const cdma::IViewPtr& ima, std::vector<int> position, void * value_ptr)
{
  memcpy(&m_data[ima->getElementOffset(position)], value_ptr, m_elem_size);
  m_dirty = true;
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::deepCopy
//----------------------------------------------------------------------------
template<typename T> IArrayStoragePtr DefaultArrayStorage<T>::deepCopy()
{
  T* data = new T[m_array_length];
  memcpy( data, m_data, m_array_length * sizeof(T) );
  return IArrayStoragePtr(new DefaultArrayStorage( data, m_array_length ));
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::deepCopy(view)
//----------------------------------------------------------------------------
template<typename T> IArrayStoragePtr DefaultArrayStorage<T>::deepCopy(IViewPtr view)
{
  CDMA_FUNCTION_TRACE("DefaultArrayStorage<T>::deepCopy(IViewPtr view)");
  // Initialize memory
  T* data = new T[view->getSize()];
  T* storage = data;

  // Init start, pos & size of buff
  int rank = view->getRank();
  unsigned int length;
  unsigned int nbBytes;
  int startSrc = 0;
  int startDst = 0;
  int current  = 0;
  int last     = 1;
  std::vector<int> shape = view->getShape();
  std::vector<int> stride = view->getStride();
  std::vector<int> position ( shape.size() );
 
  // If two consecutive cells are adjacents
  if( abs(stride[ rank - 1 ]) == 1 )
  {
    // Data will be copied slabs of shape[rank-1] length
    length  = view->getShape()[ rank - 1 ];
    nbBytes = length * sizeof( T );

    // Only consider rank - 1 first dimensions for iterations
    shape.pop_back();
    position[ rank - 1 ] = 0;
    rank = rank - 1;
  }
  else
  {
    // Nothing to do: the array will be copied point by point
    length  = 1;
    nbBytes = length * sizeof( T );
    position[ rank - 1 ] = 0;
  }
  
  // Calculate last index position of slice
  for( int i = 0; i < rank; i++ )
  {
    last *= shape[i];
  }
  
  // For each slice of rank 1
  while( current <= last && current >= 0 )
  {
    // Increment position of the slice
    if( position[0] < shape[0] )
    {
      // Determine start offset in memory
      startSrc = view->getElementOffset( position );

      
      // Copy memory as it is seen according to the view
      memcpy( (void*) (data + startDst), (const void*) (m_data + startSrc), nbBytes );

      for( int i = rank - 1; i >= 0; i-- )
      {
        if( position[i] + 1 >= shape[i] && i > 0)
        {
          position[i] = 0;
        }
        else
        {
          position[i]++;
          break;
        }
      }
      // Update next start step in destination storage
      startDst += length;
    }
    else
    {
      break;
    }
    
    current++;
  }
  return IArrayStoragePtr(new DefaultArrayStorage<T>( storage, view->getSize()));
}

}

/// @endcond

#endif // __CDMA_DEFAULTARRAYSTORAGE_HPP__
