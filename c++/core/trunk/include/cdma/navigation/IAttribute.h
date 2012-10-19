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

#ifndef __CDMA_IATTRIBUTE_H__
#define __CDMA_IATTRIBUTE_H__

#include <typeinfo>

#include <cdma/array/IArray.h>
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

  /// True if value is an array (getLength() > 1).
  ///
  /// @return true or false
  ///
  virtual bool isArray() = 0;

  /// Get the size of the array of values; = 1 if scaler.
  ///
  /// @return integer value
  ///
  virtual int getSize() = 0;

  /// Read all the data for this Attribute and return a memory resident Array.
  /// The Array has the same element type and shape as the Attribute.
  ///
  /// @return the requested data in a memory-resident Array.
  ///
  virtual IArrayPtr getData() = 0;

  /// Set the given array as new data for this IDataItem
  ///
  /// @param array IArrayPtr object
  ///
  virtual void setData(const IArrayPtr&) = 0;

  /// Convenience non-abstract method allowing to get a scalar value
  ///
  template<typename T> T getValue() 
  {
    return getData()->getValue<T>();
  }

  /// Convenience non-abstract method allowing to set a scalar value
  ///
  template<typename T> void setValue(T value) 
  {
    return setData(new Array(value));
  }
};

DECLARE_SHARED_PTR(IAttribute);

/// CDMA types
typedef std::list<IAttributePtr> AttributeList;

} //namespace cdma

#endif //__CDMA_IATTRIBUTE_H__

