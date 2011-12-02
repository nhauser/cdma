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

#include <cdma/array/IArrayIterator.h>

namespace cdma
{

//==============================================================================
/// Array iterator default implementation
//==============================================================================
class ArrayIterator : public IArrayIterator
{
private:
  std::string m_factory; // Factory name
  
public:
  /// Return true if there are more elements in the iteration.
  ///
  /// @return true or false
  ///
  bool hasNext();

  /// Return true if there is an element in the current iteration.
  ///
  /// @return true or false
  ///
  bool hasCurrent();

  /// Get next value as a double.
  ///
  /// @return double value
  ///
  double getDoubleNext();

  /// Set next value with a double.
  ///
  /// @param val double value
  ///
  void setDoubleNext(double val);

  /// Get current value as a double.
  ///
  /// @return double value
  ///
  double getDoubleCurrent();

  /// Set current value with a double.
  ///
  /// @param val double value
  ///
  void setDoubleCurrent(double val);

  /// Get next value as a float.
  ///
  /// @return float value
  ///
  float getFloatNext();

  /// Set next value with a float.
  ///
  /// @param val float value
  ///
  void setFloatNext(float val);

  /// Get current value as a float.
  ///
  /// @return float value
  ///
  float getFloatCurrent();

  /// Set current value with a float.
  ///
  /// @param val float value
  ///
  void setFloatCurrent(float val);

  /// Get next value as a long.
  ///
  /// @return long value
  ///
  long getLongNext();

  /// Set next value with a long.
  ///
  /// @param val long value
  ///
  void setLongNext(long val);

  /// Get current value as a long.
  ///
  /// @return long value
  ///
  long getLongCurrent();

  /// Set current value with a long.
  ///
  /// @param val long value
  ///
  void setLongCurrent(long val);

  /// Get next value as a int.
  ///
  /// @return integer value
  ///
  int getIntNext();

  /// Set next value with a int.
  ///
  /// @param val integer value
  ///
  void setIntNext(int val);

  /// Get current value as a int.
  ///
  /// @return integer value
  ///
  int getIntCurrent();

  /// Set current value with a int.
  ///
  /// @param val integer value
  ///
  void setIntCurrent(int val);

  /// Get next value as a short.
  ///
  /// @return short value
  ///
  short getShortNext();

  /// Set next value with a short.
  ///
  /// @param val short value
  ///
  void setShortNext(short val);

  /// Get current value as a short.
  ///
  /// @return short value
  ///
  short getShortCurrent();

  /// Set current value with a short.
  ///
  /// @param val short value
  ///
  void setShortCurrent(short val);

  /// Get next value as a byte.
  ///
  /// @return unsigned char value
  ///
  unsigned char getByteNext();

  /// Set next value with a byte.
  ///
  /// @param val unsigned char value
  ///
  void setByteNext(unsigned char val);

  /// Get current value as a byte.
  ///
  /// @return unsigned char value
  ///
  unsigned char getByteCurrent();

  /// Set current value with a byte.
  ///
  /// @param val unsigned char value
  ///
  void setByteCurrent(unsigned char val);

  /// Get next value as a char.
  ///
  /// @return char value
  ///
  char getCharNext();

  /// Set next value with a char.
  ///
  /// @param val char value
  ///
  void setCharNext(char val);

  /// Get current value as a char.
  ///
  /// @return char value
  ///
  char getCharCurrent();

  /// Set current value with a char.
  ///
  /// @param val char value
  ///
  void setCharCurrent(char val);

  /// Get next value as a bool.
  ///
  /// @return true or false
  ///
  bool getBooleanNext();

  /// Set next value with a bool.
  ///
  /// @param val true or false
  ///
  void setBooleanNext(bool val);

  /// Get current value as a bool.
  ///
  /// @return true or false
  ///
  bool getBooleanCurrent();

  /// Set current value with a bool.
  ///
  /// @param val bool true or false
  ///
  void setBooleanCurrent(bool val);

  /// Get next value as an Object.
  ///
  /// @return Object
  ///
  yat::Any getObjectNext();

  /// Set next value with a Object.
  ///
  /// @param val any Object
  ///
  void setObjectNext(const yat::Any& val);

  /// Get current value as a Object.
  ///
  /// @return Object
  ///
  yat::Any getObjectCurrent();

  /// Set current value with a Object.
  ///
  /// @param val any Object
  ///
  void setObjectCurrent(const yat::Any& val);

  /// Get next value as an Object.
  ///
  /// @return any Object
  ///
  yat::Any next();

  /// Get the current counter, use for debugging.
  ///
  /// @return array of integer
  ///
  std::vector<int> getCurrentCounter();

  //@{ IObject interface
  CDMAType::ModelType getModelType() const { return CDMAType::Other; };
  std::string getFactoryName() const { return m_factory; };
  //@}
};

}
#endif
