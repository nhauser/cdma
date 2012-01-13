//*****************************************************************************
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
//*****************************************************************************
#ifndef __CDMA_ARRAYITERATOR_H__
#define __CDMA_ARRAYITERATOR_H__

#include <vector>
#include <cdma/array/Array.h>
#include <cdma/array/View.h>

namespace cdma
{

//==============================================================================
/// Array iterator default implementation
//==============================================================================
class ArrayIterator
{
private:
  ArrayPtr         m_array;    // array storing data to iterate over
  ViewPtr         m_view;    // view of how to iterate over the array
  std::vector<int> m_position; // position in the current view of the iterator
  
public:
  ~ArrayIterator();
  ArrayIterator(const ArrayIterator& iter);
  ArrayIterator(const cdma::ArrayPtr& array, const cdma::ViewPtr& view, std::vector<int> position);

  ArrayIterator& operator++(void); // prefix operator
  ArrayIterator& operator++(int); // suffix operator
  yat::Any& operator*(void) const;
  bool operator==(const ArrayIterator& it);
  bool operator!=(const ArrayIterator& it);
  
  /// Get the current position.
  ///
  /// @return array of integer
  ///
  std::vector<int> getPosition() const;

  /// Get the current position, in a flat representation of the array
  ///
  /// @return int value
  ///  
  long currentElement() const;

  //@{ IObject interface
  CDMAType::ModelType getModelType() const { return CDMAType::Other; };
  std::string getFactoryName() const { return m_array->getFactoryName(); };
  //@}
  
  protected:
    std::vector<int>& incrementPosition(const ViewPtr& view, std::vector<int>& position);
    
    /// Get next value as an Any.
    ///
    /// @return any Any
    ///
    yat::Any next();
};

}
#endif
