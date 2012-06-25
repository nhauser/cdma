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

#include <cdma/array/Range.h>

/// @cond clientAPI

namespace cdma
{

// Forward declaration
DECLARE_CLASS_SHARED_PTR(View);

//==============================================================================
/// @brief Describes a way of parsing a array
///
/// A View is compound of ranges that describes each dimension 
/// (length, offset between to consecutive elements, offset of the first element)
///
/// The view can be used to describe a whole array, a part of the array or a 
/// different way to parse it.
//==============================================================================
class CDMA_DECL View
{
public:
  /// c-tor
  View();

  /// Copy constructor
  ///
  /// @param view the source view
  ///
  View(const cdma::ViewPtr& view );

  /// c-tor
  ///
  /// @param rank Dimensions count
  /// @param shape Length of each dimension
  /// @param start Start position in each dimension
  /// @todo add a method with stride parameter
  ///
  View(int rank, int shape[], int start[]);

  /// c-tor
  ///
  /// @param shape Length of each dimension
  /// @param start Start position in each dimension
  /// @todo remove it (see below)
  ///
  View(std::vector<int> shape, std::vector<int> start);

  /// c-tor
  ///
  /// @param shape Length of each dimension
  /// @param start Start position in each dimension
  /// @param stride offset between to consecutive element for the corresponding dimension
  /// @todo stride default value
  ///
  View(std::vector<int> shape, std::vector<int> start, std::vector<int> stride);
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
  /// @param origin An array of integers
  ///
  void setOrigin(std::vector<int> origin);
  
  /// Set the given shape for this view
  ///
  /// @param shape An array of integers
  ///
  void setShape(std::vector<int> shape);
  
  /// Set the stride for this view. The stride is the number of
  /// cells between two consecutive cells in the same dimension.
  ///
  /// @param stride An array of integers
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
  
  void compose(const ViewPtr& higher_view);
  
private:
  int                m_rank;      // Rank of the View
  std::vector<Range> m_ranges;    // Ranges that constitute the global View view in each dimension
  bool               m_upToDate;  // Does the overall shape has changed
  int                m_lastIndex; // Last indexed cell of the view 
  ViewPtr            m_compound;  // Higher View from which this one is a sub-part
  
private:
  /// Get the position vector of the element at the given offset.
  ///
  /// @return vector<int> position of the offset
  ///
  std::vector<int> getPositionElement(long offset);
};

}

/// @endcond

#endif // __CDMA_VIEW_H__
