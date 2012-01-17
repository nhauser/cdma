// *****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez Clément
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// *****************************************************************************
#ifndef __CDMA_VIEW_H__
#define __CDMA_VIEW_H__

#include <string>
#include <vector>
#include <list>

#include <cdma/array/Range.h>

namespace cdma
{

//==============================================================================
/// View of an array
/// Views describe an array : its rank, in each dimension (Range) is provided 
/// its length, offset between to consecutive elements, offset of the 
/// first element.
/// The view can be used to describe a whole array, a part of the array or a 
/// different way to consider it.
//==============================================================================
class View
{
public:
  // Constructors
  View(const cdma::ViewPtr& view );
  View(int rank, int shape[], int start[]);
  ~View();

  /// Get the number of dimensions in the array.
  ///
  /// @return integer value
  ///
  int getRank();
  
  /// Get the shape: length of array in each dimension.
  ///
  /// @return array of integer
  ///
  std::vector<int> getShape();
  
  /// Get the origin: first index of array in each dimension.
  ///
  /// @return array of integer
  ///
  std::vector<int> getOrigin();
  
  /// Get the total number of elements in the array.
  ///
  /// @return long value
  ///
  long getSize();
  
  /// Get the stride: for each dimension number elements to jump in the array between two
  /// consecutive element of the same dimension
  ///
  /// @return array of integer representing the stride
  ///
  std::vector<int> getStride();

  /// Get the index of the element pointed by the position vector
  /// projected into a 1D backing array.
  ///
  /// @return long value
  ///
  long getElementOffset(std::vector<int> position);
  
  /// Get the last element's index projected into the 1D backing array.
  ///
  /// @return integer value
  ///
  long lastElement();
  
  /// Set the origin on each dimension for this view
  ///
  /// @param origin array of integers
  ///
  void setOrigin(std::vector<int> origin);
  
  /// Set the given shape for this view
  ///
  /// @param value array of integers
  ///
  void setShape(std::vector<int> shape);
  
  /// Set the stride for this view. The stride is the number of
  /// cells between two consecutive cells in the same dimension.
  ///
  /// @param stride array of integers
  ///
  void setStride(std::vector<int> stride);
  
  /// Set the name of one of the dimension.
  ///
  /// @param dim which dimension
  /// @param name of the indexed dimension
  ///
  void setViewName(int dim, const std::string& name);
  
  /// Get the name of one of the indices.
  ///
  /// @param dim which dimension
  /// @return name of dimension, or null if none.
  ///
  std::string getViewName(int dim);
  
  /// Remove all index with length one.
  ///
  void reduce();
  
  /// Eliminate the specified dimension.
  ///
  /// @param dim dimension to eliminate: must be of length one, else send an excetpion
  ///
  void reduce(int dim) throw ( Exception );
  
private:
  int                m_rank;       // Rank of the View
  std::vector<Range> m_ranges;     // Ranges that constitute the global View view in each dimension
  bool               m_upToDate;   // Does the overall shape has changed
  int                m_lastIndex;  // last indexed cell of the view 

};

}
#endif // __CDMA_VIEW_H__
