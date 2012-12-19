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

#ifndef __CDMA_VIEW_H__
#define __CDMA_VIEW_H__

#include <cdma/array/impl/Range.h>
#include <cdma/array/IView.h>

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
class View: public IView
{
public:
  /// c-tor
  View();

  /// Copy constructor
  ///
  /// @param view the source view
  ///
  View(const cdma::IViewPtr& view );

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
 
  /// d-tor
  ~View();

  //@{ IView interface -------------

  int getRank();
  std::vector<int> getShape();
  std::vector<int> getOrigin();
  long getSize();
  std::vector<int> getStride();
  long getElementOffset(std::vector<int> position);
  long lastElement();
  void setOrigin(std::vector<int> origin);
  void setShape(std::vector<int> shape);
  void setStride(std::vector<int> stride);
  void setViewName(int dim, const std::string& name);
  std::string getViewName(int dim);
  void reduce();
  void reduce(int dim) throw ( Exception );
  void compose(const IViewPtr& higher_view);

  //@}
  
private:
  int                m_rank;      // Rank of the View
  std::vector<Range> m_ranges;    // Ranges that constitute the global View view in each dimension
  bool               m_upToDate;  // Does the overall shape has changed
  int                m_lastIndex; // Last indexed cell of the view 
  ViewPtr            m_compound;  // Higher View from which this one is a sub-part
  
  // Get the position vector of the element at the given offset.
  //
  // @return vector<int> position of the offset
  //
  std::vector<int> getPositionElement(long offset);
};

}

/// @endcond

#endif // __CDMA_VIEW_H__
