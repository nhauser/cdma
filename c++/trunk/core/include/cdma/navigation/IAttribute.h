// ******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
// ******************************************************************************
#ifndef __CDMA_IATTRIBUTE_H__
#define __CDMA_IATTRIBUTE_H__

#include <yat/utils/String.h>
#include <yat/memory/SharedPtr.h>
#include <typeinfo>

#include <cdma/Common.h>

namespace cdma
{

//==============================================================================
/// Interface IAttribute, with name and value. 
/// The metadata for data items and groups.
//==============================================================================
class CDMA_DECL IAttribute
{
public:
  virtual ~IAttribute()
  {
  }

  /// Get the name of this Attribute. Attribute names are unique within a
  /// NetcdfFile's global set, and within a Variable's set.
  ///
  /// @return string object
  ///
  virtual std::string getName() = 0;

  /// Get the data type of the Attribute value.
  ///
  /// @return Class object
  ///
  virtual const std::type_info& getType() = 0;

  /// True if value is a string or String[].
  ///
  /// @return true or false
  ///
  virtual bool isString() = 0;

  /// True if value is an array (getLength() > 1).
  ///
  /// @return true or false
  ///
  virtual bool isArray() = 0;

  /// Get the length of the array of values; = 1 if scaler.
  ///
  /// @return integer value
  ///
  virtual int getLength() = 0;

  /// Get the value as an Array.
  ///
  /// @return Array of values.
  ///
  //virtual IArrayPtr getValue() = 0;

  /// Retrieve string value; only call if isString() is true.
  ///
  /// @return string if this is a string valued attribute, else null.
  /// @see IAttribute#isString
  ///
  virtual std::string getStringValue() = 0;

  /// Retrieve string value; only call if isString() is true.
  ///
  /// @param index
  ///            integer value
  /// @return string if this is a string valued attribute, else null.
  /// @see IAttribute#isString
  ///
  //virtual std::string getStringValue(int index) = 0;

  /// Retrieve integer value
  ///
  /// @return the first element of the value array, or null if its a String.
  ///
  virtual long getIntValue() = 0;

  /// Retrieve integer value
  ///
  /// @return the first element of the value array, or null if its a String.
  ///
  virtual double getFloatValue() = 0;

  /// Retrieve a numeric value by index. If its a String, it will try to parse
  /// it as a double.
  ///
  /// @param index the index into the value array.
  /// @return double <code>value[index]</code>, or null if its a non-parsable
  ///         string or the index is out of range.
  ///
  //virtual double getNumericValue(int index) = 0;

  /// string representation.
  ///
  /// @return string object
  ///
  virtual std::string toString() = 0;

  /// set the value as a String, trimming trailing zeroes.
  ///
  /// @param val string object
  ///
  virtual void setStringValue(const std::string& val) = 0;

  /// set the values from an Array.
  ///
  /// @param value IArray object
  ///
  virtual void setValue(const yat::Any& value) = 0;
};

DECLARE_SHARED_PTR(IAttribute);

} //namespace CDMACore
#endif //__CDMA_IATTRIBUTE_H__

