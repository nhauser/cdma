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

#ifndef __CDMA_IVIEW_H__
#define __CDMA_IVIEW_H__

/// @cond clientAPI

namespace cdma
{

// Forward declaration
DECLARE_CLASS_SHARED_PTR(IView);

//==============================================================================
/// @brief Describes a way of parsing a array
///
/// A View is compound of ranges that describes each dimension 
/// (length, offset between to consecutive elements, offset of the first element)
///
/// The view can be used to describe a whole array, a part of the array or a 
/// different way to parse it.
//==============================================================================
class CDMA_DECL IView
{
public:

  /// d-tor
  ///
  virtual ~IView() {}

  /// Get the number of dimensions in the array.
  ///
  /// @return integer value
  ///
  virtual int getRank() = 0;
  
  /// Get the shape: length of array in each dimension.
  ///
  /// @return array of integer
  ///
  virtual std::vector<int> getShape() = 0;
  
  /// Get the origin: first index of array in each dimension.
  ///
  /// @return array of integer
  ///
  virtual std::vector<int> getOrigin() = 0;
  
  /// Get the total number of elements in the array.
  ///
  /// @return long value
  ///
  virtual long getSize() = 0;
  
  /// Get the stride: for each dimension number elements to jump in the array between two
  /// consecutive element of the same dimension
  ///
  /// @return array of integer representing the stride
  ///
  virtual std::vector<int> getStride() = 0;

  /// Get the index of the element pointed by the position vector
  /// projected into a 1D backing array.
  ///
  /// @return long value
  ///
  virtual long getElementOffset(std::vector<int> position) = 0;
  
  /// Get the last element's index projected into the 1D backing array.
  ///
  /// @return integer value
  ///
  virtual long lastElement() = 0;
  
  /// Set the origin on each dimension for this view
  ///
  /// @param origin An array of integers
  ///
  virtual void setOrigin(std::vector<int> origin) = 0;
  
  /// Set the given shape for this view
  ///
  /// @param shape An array of integers
  ///
  virtual void setShape(std::vector<int> shape) = 0;
  
  /// Set the stride for this view. The stride is the number of
  /// cells between two consecutive cells in the same dimension.
  ///
  /// @param stride An array of integers
  ///
  virtual void setStride(std::vector<int> stride) = 0;
  
  /// Set the name of one of the dimension.
  ///
  /// @param dim which dimension
  /// @param name of the indexed dimension
  ///
  virtual void setViewName(int dim, const std::string& name) = 0;
  
  /// Get the name of one of the indices.
  ///
  /// @param dim which dimension
  /// @return name of dimension, or null if none.
  ///
  virtual std::string getViewName(int dim) = 0;
  
  /// Remove all index with length one.
  ///
  virtual void reduce() = 0;
  
  /// Eliminate the specified dimension.
  ///
  /// @param dim dimension to eliminate: must be of length one, else send an excetpion
  ///
  virtual void reduce(int dim) throw ( Exception ) = 0;
  
  virtual void compose(const IViewPtr& higher_view) = 0;

private:
  IView() {}
  
public:
  // implementation
  friend class View;
};

}

/// @endcond

#endif // __CDMA_IVIEW_H__
