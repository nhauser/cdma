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

#ifndef __CDMA_IDATAITEM_H__
#define __CDMA_IDATAITEM_H__

// Standard includes
#include <vector>
#include <typeinfo>

// CDMA includes
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IContainer.h>
#include <cdma/navigation/IGroup.h>
#include <cdma/navigation/IDimension.h>
#include <cdma/array/IArray.h>

namespace cdma
{

//==============================================================================
/// @brief Abstraction of a data item container.
///
/// It has a DataType, a set of Dimensions that define its Array shape,
/// and optionally a set of Attributes (see IAttribute).
///
/// @note It must be a base class of data format engine data item class concretization
/// which can be overrided, if needed, by plug-ins based on same engine
//==============================================================================
class CDMA_DECL IDataItem : public IContainer 
{
public:
  //Virtual destructor
  virtual ~IDataItem() {}

  /// Find an Attribute by name, ignoring the case.
  ///
  /// @param name the name of the attribute
  /// @return the attribute, or null if not found
  ///
  virtual IAttributePtr findAttributeIgnoreCase(const std::string& name) = 0;

  /// Find the index of the named Dimension in this DataItem.
  ///
  /// @param name the name of the dimension
  /// @return the index of the named Dimension, or -1 if not found.
  ///
  virtual int findDimensionView(const std::string& name) = 0;

  /// Get its parent Group, or null if its the root group.
  ///
  /// @return IGroup object
  ///
  virtual IGroupPtr getParent() = 0;

  /// Get the root group of the tree that holds the current Group.
  ///
  /// @return IGroup object
  ///
  virtual IGroupPtr getRoot() = 0;

  /// Read all the data for this DataItem and return a memory resident Array.
  /// The Array has the same element type and shape as the DataItem.
  ///
  /// @return the requested data in a memory-resident Array.
  ///
  virtual IArrayPtr getData(std::vector<int> position = std::vector<int>() ) throw ( Exception ) = 0;

  /// Read a section of the data for this DataItem and return a memory resident
  /// Array. The Array has the same element type as the DataItem. The size of
  /// the Array will be either smaller or equal to the DataItem.
  ///
  /// @param origin array of int
  /// @param shape array of int
  /// @return the requested data in a memory-resident Array.
  ///
  virtual IArrayPtr getData( std::vector<int> origin, std::vector<int> shape) throw ( Exception ) = 0;

  /// Get data as scalar value
  ///
  /// @return value
  ///
  template<typename T> T getValue(std::vector<int> position = std::vector<int>())
  {
      return this->getData(position)->getValue<T>();
  }

  ///
  /// Get the description of the DataItem. Default is to use "long_name"
  /// attribute value. If not exist, look for "description", "title", or
  /// "standard_name" attribute value (in that order).
  ///
  /// @return description, or null if not found.
  ///
  virtual std::string getDescription() = 0;

  /// Get the ith dimensions (if several are available return a populated corresponding list).
  ///
  /// @param i index of the dimension.
  /// @return requested Dimensions, or null if i is out of bounds.
  ///
  virtual std::list<IDimensionPtr > getDimensions(int i) = 0;

  /// Get the list of all dimensions used by this variable. The most slowly varying
  /// (leftmost for Java and C programmers) dimension is first. For scalar
  /// variables, the list is empty.
  ///
  /// @return list with objects of type ucar.nc2.Dimension
  ///
  virtual std::list<IDimensionPtr > getDimensionList() = 0;

  /// Get the list of Dimension names, space delineated.
  ///
  /// @return string object
  ///
  virtual std::string getDimensionsString() = 0;

  /// Get the number of bytes for one element of this DataItem. For DataItems
  /// of primitive type, this is equal to getDataType().getSize(). DataItems of
  /// string type does not know their size, so what they return is undefined.
  /// DataItems of Structure type return the total number of bytes for all the
  /// members of one Structure, plus possibly some extra padding, depending on
  /// the underlying format. DataItems of Sequence type return the number of
  /// bytes of one element.
  ///
  /// @return total number of bytes for the DataItem
  ///
  virtual int getElementSize() = 0;

  /// Get the number of dimensions of the DataItem.
  ///
  /// @return integer value
  ///
  virtual int getRank() = 0;

  /// Get the shape: length of DataItem in each dimension.
  ///
  /// @return int array whose length is the rank of this and whose values equal
  ///        the length of that Dimension.
  ///
  virtual std::vector<int> getShape() = 0;

  /// Get the total number of elements in the DataItem. If this is an unlimited
  /// DataItem, will return the current number of elements. If this is a
  /// Sequence, will return 0.
  ///
  /// @return total number of elements in the DataItem.
  ///
  virtual long getSize() = 0;

  /// Create a new DataItem that is a logical slice of this DataItem, by fixing
  /// the specified dimension at the specified index value. This reduces rank
  /// by 1. No data is read until a read method is called on it.
  ///
  /// @param dim which dimension to fix
  /// @param value at what index value
  /// @return a new DataItem which is a logical slice of this DataItem.
  ///
  virtual IDataItemPtr getSlice(int dim, int value) throw ( Exception ) = 0;

  /// Get the java class of the DataItem data.
  ///
  /// @return Class object
  ///
  virtual const std::type_info& getType() = 0;

  /// Get the Unit string for the DataItem. Default is to use "units" attribute
  /// value
  ///
  /// @return unit string, or null if not found.
  ///
  virtual std::string getUnit() = 0;

  /// Whether this is a scalar DataItem (rank == 0).
  ///
  /// @return true or false
  ///
  virtual bool isScalar() = 0;

  /// Can this variable's size grow?. This is equivalent to saying at least one
  /// of its dimensions is unlimited.
  ///
  /// @return bool true iff this variable can grow
  ///
  virtual bool isUnlimited() = 0;

  /// Is this DataItem unsigned?. Only meaningful for byte, short, int, long
  /// types.
  ///
  /// @return true or false
  ///
  virtual bool isUnsigned() = 0;

  /// Remove an Attribute : uses the attribute hashCode to find it.
  ///
  /// @param attr IAttribute object
  /// @return true if was found and removed
  ///
  virtual bool removeAttribute(const IAttributePtr& attr) = 0;

  /// Set the data type.
  ///
  /// @param dataType type_info of the underlying data
  ///
  virtual void setDataType(const std::type_info& dataType) = 0;
  
  /// Set the given array as new data for this IDataItem
  ///
  /// @param array IArrayPtr object
  ///
  virtual void setData(const IArrayPtr& array ) = 0;

  /// Set the dimension on the specified index.
  ///
  /// @param dim_ptr IDimension to add to this data item
  /// @param index Index the dimension matches
  ///
  virtual void setDimension(const IDimensionPtr& dim_ptr, int index) throw ( Exception ) = 0;

  /// Set the units of the DataItem.
  ///
  /// @param units string object Created on 20/03/2008
  ///
  virtual void setUnit(const std::string& units) = 0;

 };
 
 typedef std::list<IDataItemPtr> DataItemList;

} //namespace cdma

#endif //__CDMA_IDATAITEM_H__

