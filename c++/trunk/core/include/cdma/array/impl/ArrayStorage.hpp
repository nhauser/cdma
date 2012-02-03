//*****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez ClÃ©ment
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
//*****************************************************************************
#ifndef __CDMA_DEFAULTARRAYSTORAGE_HPP__
#define __CDMA_DEFAULTARRAYSTORAGE_HPP__

#include <string.h>

namespace cdma
{

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::~DefaultArrayStorage
//----------------------------------------------------------------------------
template<typename T> DefaultArrayStorage<T>::~DefaultArrayStorage()
{
    CDMA_FUNCTION_TRACE("DefaultArrayStorage::~DefaultArrayStorage");
    delete m_data;
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::DefaultArrayStorage
//----------------------------------------------------------------------------
template<typename T> DefaultArrayStorage<T>::DefaultArrayStorage( T* data, size_t length )
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
  for( int i = 0; i < shape.size(); i++ )
  {
    m_array_length *= shape[i];
  }
  m_dirty = false;
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::set
//----------------------------------------------------------------------------
template<typename T> yat::Any& DefaultArrayStorage<T>::get( const cdma::ViewPtr& view, std::vector<int> position )
{
  long idx = view->getElementOffset(position);
  m_current = m_data[idx];
  return m_current;
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::set
//----------------------------------------------------------------------------
template<typename T> void DefaultArrayStorage<T>::set(const cdma::ViewPtr& ima, std::vector<int> position, const yat::Any& value)
{
  m_data[ima->getElementOffset(position)] = yat::any_cast<T>(value);
  m_dirty = true;
}

//----------------------------------------------------------------------------
// DefaultArrayStorage<T>::set
//----------------------------------------------------------------------------
template<typename T> IArrayStoragePtr DefaultArrayStorage<T>::deepCopy()
{
  T* data = new T[m_array_length];
  memcpy( data, m_data, m_array_length * sizeof(T) );
  return new DefaultArrayStorage( data, m_array_length );
}

template<typename T> IArrayStoragePtr DefaultArrayStorage<T>::deepCopy(ViewPtr view)
{
  // Initialize memory
  T* data = new T[view->getSize()];
  T* storage = data;

  // Init start, pos & size of buff
  unsigned int length  = view->getShape()[ view->getRank() - 1 ];
  unsigned int nbBytes = length * sizeof( T );
  int startSrc = 0;
  int startDst = 0;
  int current  = 0;
  int last     = 1;
  std::vector<int> shape = view->getShape();
  std::vector<int> position ( shape.size() );
  
  // Only consider rank - 1 first dimensions
  shape.pop_back();
  position[shape.size() - 1] = 0;
  
  // Calculate last index position of slice
  for( int i = 0; i < shape.size(); i++ )
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

      // Copy memory
      memcpy( (void*) (data + startDst), (const void*) (m_data + startSrc), nbBytes );

      for( unsigned int i = shape.size() - 1; i >= 0; i-- )
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
  
  return new DefaultArrayStorage<T>( storage, view->getSize() );
}

}


#endif // __CDMA_DEFAULTARRAYSTORAGE_HPP__
