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

#include <yat/any/Any.h>

#include <cdma/IObject.h>
#include <cdma/IObject.h>

namespace cdma
{

//==============================================================================
/// Array iterator to make transversal parsing of the Array
//==============================================================================

class ArrayIterator
{
public:
  // Constructor
  ArrayIterator(const ArrayIterator& iter);
  ArrayIterator(const cdma::ArrayPtr& array, const cdma::ViewPtr& view, std::vector<int> position);

  // d-structor
  ~ArrayIterator();

  /// Prefix operator: increments the ArrayIterator before
  /// returning the reference of the next indexed element
  ///
  ArrayIterator& operator++(void);
  
  /// Suffix operator: increments the ArrayIterator after having 
  /// returned the reference of the current indexed element
  ///
  ArrayIterator operator++(int);

  /// Prefix operator: decrements the ArrayIterator before
  /// returning the reference of the previous indexed element
  ///
  ArrayIterator& operator--(void);
  
  /// Suffix operator: decrements the ArrayIterator after having 
  /// returned the reference of the current indexed element
  ///
  ArrayIterator operator--(int);
  
  /// Access method with explicit value conversion
  ///
  /// @return value converted to the type T if possible
  ///
  template <typename T> T getValue(void) const { return m_array->getValue<T>(m_view, m_position); }
  
  /// Comparison operator: egality
  ///
  /// @param iterator to compare this
  /// @return true if both iterator refers to the same cell index
  ///
  bool operator==(const ArrayIterator& iterator);
  
  /// Comparison operator: difference
  ///
  /// @param iterator to compare this
  /// @return true if both iterator refers to different cell index
  ///
  bool operator!=(const ArrayIterator& iterator);
  
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
    /// Increments the position vector according the defined view
    ///
    /// @param view used to describe how to increments
    /// @param position to be incremented
    ///
    std::vector<int>& incrementPosition(const ViewPtr& view, std::vector<int>& position);

    /// Decrements the position vector according the defined view
    ///
    /// @param view used to describe how to increments
    /// @param position to be incremented
    ///
    std::vector<int>& decrementPosition(const ViewPtr& view, std::vector<int>& position);
    
    /// Returns true if the position is equivalent to the last position
    /// that iterator can take
    ///
    bool isEndPosition(std::vector<int> shape, std::vector<int> position);

    /// Returns true if the position is equivalent to the first position
    /// that iterator can take
    ///    
    bool isBeginPosition(std::vector<int> shape, std::vector<int> position);
    
private:
  ArrayPtr         m_array;    // array storing data to iterate over
  ViewPtr          m_view;     // view of how to iterate over the array
  std::vector<int> m_position; // position in the current view of the iterator
};

}
#endif
