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

#ifndef __CDMA_IDATA_H__
#define __CDMA_IDATA_H__

// std
#include <vector>

// cdma
#include <cdma/Common.h>
#include <cdma/array/IView.h>

/// @cond engineAPI

namespace cdma
{

// Forward declaration
DECLARE_CLASS_SHARED_PTR(IArrayStorage);

//==============================================================================
/// @brief Abstraction of the physical container of the array matrix.
//==============================================================================
class CDMA_DECL IArrayStorage
{
public:

  /// Get pointer to the "value" from the memory buffer according the position in the given view.
  ///
  /// @param view_ptr Shared pointer on the view to consider for the index calculation
  /// @param position into which the value will be set
  /// @return anonymous pointer to the value
  ///
  virtual void *getValue( const cdma::IViewPtr& view_ptr, std::vector<int> position ) = 0;

  /// Set "value" in the memory buffer according the position in the given view. The 
  /// given yat::Any will be casted into memory buffer type.
  ///
  /// @param view_ptr Shared pointer on the view to consider for the index calculation
  /// @param position into which the value will be set
  /// @param value_ptr C-style pointer to memory position to be set
  ///
  virtual void setValue(const cdma::IViewPtr& view_ptr, std::vector<int> position, void *value_ptr) = 0;

  /// Returns the type_info of the underlying canonical data
  ///
  virtual const std::type_info& getType() = 0;

  /// Returns true if the memory has been modified since last read
  ///
  virtual bool dirty() = 0;
  
  /// Set the dirty flag to given boolean.
  ///
  virtual void setDirty(bool dirty) = 0;
  
  /// Returns the underlying buffer as it is in memory
  ///
  /// @note use this method at your own risk
  ///
  virtual void* getStorage() = 0;
  
  /// Create a copy of this IArrayStorage, copying the data as it is in memory.
  ///
  /// @return the new IArrayStorage with copied memory storage
  /// @note be aware: can lead to out of memory 
  ///
  virtual IArrayStoragePtr deepCopy() = 0;

  /// Create a copy of this IArrayStorage according to the given,
  /// copying the data so that physical order is
  /// the same as logical order. Only the viewable part memory will be
  /// copied.
  ///
  /// @return the new IArrayStorage with copied memory storage
  /// @note be aware: can lead to out of memory 
  ///  
  virtual IArrayStoragePtr deepCopy(IViewPtr view) = 0;
};

/// @endcond engineAPI

}

#endif
