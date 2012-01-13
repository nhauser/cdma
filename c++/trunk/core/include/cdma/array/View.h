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
/// View default implementation
/// Views for Multidimensional Arrays. An View refers to a particular
/// element of an Array.
//==============================================================================
class View
{
private:
  int                m_rank;       // Rank of the View
  std::vector<Range> m_ranges;     // Ranges that constitute the global View view in each dimension
  bool               m_upToDate;   // Does the overall shape has changed
  int                m_lastIndex;

public:
  // Constructors
  View(const cdma::ViewPtr& view );
  View(int rank, int shape[], int start[]);
  ~View();

  //@{ View interface
  int                  getRank();
  std::vector<int>     getShape();
  std::vector<int>     getOrigin();
  long                 getSize();
  std::vector<int>     getStride();
  long                 getElementOffset(std::vector<int> position);
  long                 lastElement();
  void                 setOrigin(std::vector<int> origin);
  void                 setShape(std::vector<int> shape);
  void                 setStride(std::vector<int> stride);
  void                 setViewName(int dim, const std::string& name);
  std::string          getViewName(int dim);
  void                 reduce();
  void                 reduce(int dim) throw ( Exception );
};

}
#endif // __CDMA_VIEW_H__
