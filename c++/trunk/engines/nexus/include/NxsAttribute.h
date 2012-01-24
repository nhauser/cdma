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
#ifndef __CDMA_NXSATTRIBUTE_H__
#define __CDMA_NXSATTRIBUTE_H__

/// CDMA Attribute, with name and value. The metadata for data items and groups.

#include <string>
#include <yat/utils/String.h>
#include <nxfile.h>

#include <internal/common.h>
#include <cdma/navigation/IAttribute.h>

namespace cdma
{

//==============================================================================
/// IAttribute implementation for NeXus engine
/// See IAttribute definition for more explanation
//==============================================================================
class CDMA_DECL NxsAttribute : public IAttribute
{
private:
  NexusAttrInfo* m_info_ptr;
  void *m_value;

public:
  NxsAttribute();
  NxsAttribute( NexusFilePtr file, NexusAttrInfo* info );
  //NxsAttribute( const string& name, const string value ) { m_name = name; m_value = new string(value); };

  /// Get the name of this Attribute. Attribute names are unique within a
  /// global set, and within a Variable's set.
  ///
  /// @return string object
  ///
  std::string getName();

  /// Get the data type of the Attribute value.
  ///
  /// @return type_info
  ///
  const std::type_info& getType();

  /// True if value is a string or String[].
  ///
  /// @return true or false
  ///
  bool isString();

  /// True if value is an array (getLength() > 1).
  ///
  /// @return true or false
  ///
  bool isArray();

  /// Get the length of the array of values; = 1 if scaler.
  ///
  /// @return integer value
  ///
  int getLength();

  /// Get the value as an Array.
  ///
  /// @return Array of values.
  ///
  //ArrayPtr getValue();

  /// Retrieve string value; only call if isString() is true.
  ///
  /// @return string if this is a string valued attribute, else null.
  ///
  std::string getStringValue();

  /// Retrieve string value; only call if isString() is true.
  ///
  /// @param index integer value
  /// @return string if this is a string valued attribute, else null.
  ///
//  string getStringValue(int index);

  /// Retrieve numeric value. Equivalent to <code>getNumericValue(0)</code>
  ///
  /// @return the first element of the value array, or null if its a String.
  ///
//  double getNumericValue();

  /// Retrieve a numeric value by index. If its a String, it will try to parse
  /// it as a double.
  ///
  /// @param index the index into the value array.
  /// @return Number <code>value[index]</code>, or null if its a non-parsable
  ///         string or the index is out of range.
  ///
//  double getNumericValue(int index);

  /// string representation.
  ///
  /// @return string object
  ///
  std::string toString();

  /// set the value as a String, trimming trailing zeroes.
  ///
  /// @param val string object
  ///
  void setStringValue(std::string val);

  /// set the values from an Array.
  ///
  /// @param value Array object
  ///
  void setValue(Array& value);

  /// Retrieve entire value.
  ///
  /// @return the first element of the value array, or null if its a String.
  ///
  long getIntValue();

  /// Retrieve floating value.
  ///
  /// @return the first element of the value array, or null if its a String.
  ///
  double getFloatValue();
  
  std::string getFactoryName() const { return NXS_FACTORY_NAME; };
  CDMAType::ModelType getModelType() const { return CDMAType::Attribute; };
 };
}
#endif
