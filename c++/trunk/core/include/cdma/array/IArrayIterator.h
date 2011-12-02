//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IARRAYITERATOR_H__
#define __CDMA_IARRAYITERATOR_H__

#include <vector>

#include <yat/any/Any.h>
#include <yat/memory/SharedPtr.h>

#include <cdma/IObject.h>

namespace cdma
{

//==============================================================================
/// IArrayIterator interface
/// Array iterator...
//==============================================================================
class IArrayIterator : public IObject 
{
public:
  /// d-tor
  virtual ~IArrayIterator()
  {
  }

  /// Return true if there are more elements in the iteration.
  ///
  /// @return true or false
  ///
  virtual bool hasNext() = 0;

  /// Return true if there is an element in the current iteration.
  ///
  /// @return true or false
  ///
  virtual bool hasCurrent() = 0;

  /// Get next value as a double.
  ///
  /// @return double value
  ///
  virtual double getDoubleNext() = 0;

  /// Set next value with a double.
  ///
  /// @param val double value
  ///
  virtual void setDoubleNext(double val) = 0;

  /// Get current value as a double.
  ///
  /// @return double value
  ///
  virtual double getDoubleCurrent() = 0;

  /// Set current value with a double.
  ///
  /// @param val double value
  ///
  virtual void setDoubleCurrent(double val) = 0;

  /// Get next value as a float.
  ///
  /// @return float value
  ///
  virtual float getFloatNext() = 0;

  /// Set next value with a float.
  ///
  /// @param val float value
  ///
  virtual void setFloatNext(float val) = 0;

  /// Get current value as a float.
  ///
  /// @return float value
  ///
  virtual float getFloatCurrent() = 0;

  /// Set current value with a float.
  ///
  /// @param val float value
  ///
  virtual void setFloatCurrent(float val) = 0;

  /// Get next value as a long.
  ///
  /// @return long value
  ///
  virtual long getLongNext() = 0;

  /// Set next value with a long.
  ///
  /// @param val long value
  ///
  virtual void setLongNext(long val) = 0;

  /// Get current value as a long.
  ///
  /// @return long value
  ///
  virtual long getLongCurrent() = 0;

  /// Set current value with a long.
  ///
  /// @param val long value
  ///
  virtual void setLongCurrent(long val) = 0;

  /// Get next value as a int.
  ///
  /// @return integer value
  ///
  virtual int getIntNext() = 0;

  /// Set next value with a int.
  ///
  /// @param val integer value
  ///
  virtual void setIntNext(int val) = 0;

  /// Get current value as a int.
  ///
  /// @return integer value
  ///
  virtual int getIntCurrent() = 0;

  /// Set current value with a int.
  ///
  /// @param val integer value
  ///
  virtual void setIntCurrent(int val) = 0;

  /// Get next value as a short.
  ///
  /// @return short value
  ///
  virtual short getShortNext() = 0;

  /// Set next value with a short.
  ///
  /// @param val short value
  ///
  virtual void setShortNext(short val) = 0;

  /// Get current value as a short.
  ///
  /// @return short value
  ///
  virtual short getShortCurrent() = 0;

  /// Set current value with a short.
  ///
  /// @param val short value
  ///
  virtual void setShortCurrent(short val) = 0;

  /// Get next value as a byte.
  ///
  /// @return unsigned char value
  ///
  virtual unsigned char getByteNext() = 0;

  /// Set next value with a byte.
  ///
  /// @param val unsigned char value
  ///
  virtual void setByteNext(unsigned char val) = 0;

  /// Get current value as a byte.
  ///
  /// @return unsigned char value
  ///
  virtual unsigned char getByteCurrent() = 0;

  /// Set current value with a byte.
  ///
  /// @param val unsigned char value
  ///
  virtual void setByteCurrent(unsigned char val) = 0;

  /// Get next value as a char.
  ///
  /// @return char value
  ///
  virtual char getCharNext() = 0;

  /// Set next value with a char.
  ///
  /// @param val char value
  ///
  virtual void setCharNext(char val) = 0;

  /// Get current value as a char.
  ///
  /// @return char value
  ///
  virtual char getCharCurrent() = 0;

  /// Set current value with a char.
  ///
  /// @param val char value
  ///
  virtual void setCharCurrent(char val) = 0;

  /// Get next value as a bool.
  ///
  /// @return true or false
  ///
  virtual bool getBooleanNext() = 0;

  /// Set next value with a bool.
  ///
  /// @param val true or false
  ///
  virtual void setBooleanNext(bool val) = 0;

  /// Get current value as a bool.
  ///
  /// @return true or false
  ///
  virtual bool getBooleanCurrent() = 0;

  /// Set current value with a bool.
  ///
  /// @param val bool true or false
  ///
  virtual void setBooleanCurrent(bool val) = 0;

  /// Get next value as an Object.
  ///
  /// @return Object
  ///
  virtual yat::Any getObjectNext() = 0;

  /// Set next value with a Object.
  ///
  /// @param val any Object
  ///
  virtual void setObjectNext(const yat::Any& val) = 0;

  /// Get current value as a Object.
  ///
  /// @return Object
  ///
  virtual yat::Any getObjectCurrent() = 0;

  /// Set current value with a Object.
  ///
  /// @param val any Object
  ///
  virtual void setObjectCurrent(const yat::Any& val) = 0;

  /// Get next value as an Object.
  ///
  /// @return any Object
  ///
  virtual yat::Any next() = 0;

  /// Get the current counter, use for debugging.
  ///
  /// @return array of integer
  ///
  virtual std::vector<int> getCurrentCounter() = 0;
};

} //namespace CDMACore
#endif //__CDMA_IARRAYITERATOR_H__

