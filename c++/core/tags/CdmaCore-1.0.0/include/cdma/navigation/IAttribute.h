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

#include <typeinfo>

#include <cdma/array/Array.h>
#include <cdma/Common.h>

namespace cdma
{

//==============================================================================
/// @brief Abstraction of the metadata related to a IDataItem or a IGroup
///
/// @note It must be a base class of data format engine attribute class concretization
/// which can be overrided, if needed, by plug-ins based on same engine
//==============================================================================
class CDMA_DECL IAttribute
{
public:
  virtual ~IAttribute()
  {
  }

  /// Get the name of this Attribute that is unique within a IContainer.
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

  /// Retrieve string value; only call if isString() is true.
  ///
  /// @return string if this is a string valued attribute, else null.
  /// @see IAttribute#isString
  ///
  virtual std::string getStringValue() = 0;

  /// Retrieve integer value as long C-type
  ///
  /// @return the value as a integer (converted if needed)
  /// @throw throw an cdma::Exception if type conversion isn't possible
  ///
  virtual long getIntValue() = 0;

  /// Retrieve floating point value as double C-type
  ///
  /// @return the value as a double (converted if needed)
  /// @throw throw an cdma::Exception if type conversion isn't possible
  ///
  virtual double getFloatValue() = 0;

  /// string representation.
  ///
  /// @return string object
  ///
  virtual std::string toString() = 0;

  /// set the value as a string, trimming trailing zeroes.
  ///
  /// @param val string object
  ///
  virtual void setStringValue(const std::string& val) = 0;

  /// set the value as an integer.
  ///
  /// @param value int
  ///
  virtual void setIntValue(int value) = 0;
  
  /// set the value as a float.
  ///
  /// @param value float
  ///
  virtual void setFloatValue(float value) = 0;
};

DECLARE_SHARED_PTR(IAttribute);

/// CDMA types
typedef std::list<IAttributePtr> AttributeList;

} //namespace cdma

#endif //__CDMA_IATTRIBUTE_H__

