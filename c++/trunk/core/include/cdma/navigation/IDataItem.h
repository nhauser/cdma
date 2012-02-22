//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IDATAITEM_H__
#define __CDMA_IDATAITEM_H__

// Standard includes
#include <list>
#include <vector>
#include <typeinfo>

#include <yat/utils/String.h>
#include <yat/threading/Mutex.h>
#include <yat/memory/SharedPtr.h>

// CDMA includes
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IContainer.h>
#include <cdma/navigation/IGroup.h>
#include <cdma/navigation/IDimension.h>
#include <cdma/array/Array.h>

namespace cdma
{

//==============================================================================
/// DataItem Interface
/// A DataItem is a logical container for data. It has a DataType, a set of
/// Dimensions that define its array shape, and optionally a set of Attributes.
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
  /// @param namethe name of the dimension
  /// @return the index of the named Dimension, or -1 if not found.
  ///
  virtual int findDimensionView(const std::string& name) = 0;

  /// Create a new DataItem that is a logical slice of this DataItem, by fixing
  /// the specified dimension at the specified index value. This reduces rank
  /// by 1. No data is read until a read method is called on it.
  ///
  /// @param dimension which dimension to fix
  /// @param value at what index value
  /// @return a new DataItem which is a logical slice of this DataItem.
  ///
  virtual IDataItemPtr getASlice(int dimension, int value) throw ( Exception ) = 0;

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
  /// If the DataItem is a member of an array of Structures, this returns only
  /// the variable's data in the first Structure, so that the Array shape is
  /// the same as the DataItem. To read the data in all structures, use
  /// readAllStructures().
  ///
  /// @return the requested data in a memory-resident Array.
  ///
  virtual ArrayPtr getData(std::vector<int> position = std::vector<int>() ) throw ( Exception ) = 0;
  // Methode initiale:		  virtual ArrayPtr getData() throw ( Exception ) = 0;

  /// Read a section of the data for this DataItem and return a memory resident
  /// Array. The Array has the same element type as the DataItem. The size of
  /// the Array will be either smaller or equal to the DataItem.
  /// If the DataItem is a member of an array of Structures, this returns only
  /// the variable's data in the first Structure, so that the Array shape is
  /// the same as the DataItem. To read the data in all structures, use
  /// readAllStructures().
  ///
  /// @param origin array of int
  /// @param shape array of int
  /// @return the requested data in a memory-resident Array.
  ///
  virtual ArrayPtr getData( std::vector<int> origin, std::vector<int> shape) throw ( Exception ) = 0;
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

  /// display name plus the dimensions.
  ///
  /// @return string object
  ///
  virtual std::string getNameAndDimensions() = 0;

  /// Display name plus the dimensions.
  ///
  /// @param useFullName true or false value
  /// @param showDimLength true or false value
  ///
  virtual std::string getNameAndDimensions(bool useFullName, bool showDimLength) = 0;

  /// Get shape as an array of Range objects.
  ///
  /// @return array of Ranges, one for each Dimension.
  ///
  virtual std::list<RangePtr > getRangeList() = 0;

  /// Get the number of dimensions of the DataItem.
  ///
  /// @return integer value
  ///
  virtual int getRank() = 0;

  /// Create a new DataItem that is a logical subsection of this DataItem. No
  /// data is read until a read method is called on it.
  ///
  /// @param section
  ///           list of type Range, with size equal to getRank(). Each Range
  ///           corresponds to a Dimension, and specifies the section of data
  ///           to read in that Dimension. A Range object may be null, which
  ///           means use the entire dimension.
  /// @return a new DataItem which is a logical section of this DataItem.
  ///
  virtual IDataItemPtr getSection(std::list<RangePtr > section) throw ( Exception ) = 0;

  /// Get index subsection as an array of Range objects, relative to the
  /// original variable. If this is a section, will reflect the index range
  /// relative to the original variable. If its a slice, it will have a rank
  /// different from this variable. Otherwise it will correspond to this
  /// DataItem's shape, ie match getRanges().
  ///
  /// @return array of Ranges, one for each Dimension.
  ///
  virtual std::list<RangePtr > getSectionRanges() = 0;

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

  /// If total data is less than SizeToCache in bytes, then cache.
  ///
  /// @return integer value
  ///
  virtual int getSizeToCache() = 0;

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
  virtual std::string getUnitsString() = 0;

  ///
  /// Does this have its data read in and cached?
  ///
  /// @return true or false
  ///
  virtual bool hasCachedData() = 0;

  ///
  /// Override Object.hashCode() to implement equals.
  ///
  /// @return integer value
  ///
  //##virtual int hashCode() = 0;

  ///
  /// Invalidate the data cache.
  ///
  virtual void invalidateCache() = 0;

  /// Will this DataItem be cached when read. Set externally, or calculated
  /// based on total size < sizeToCache.
  ///
  /// @return true is caching
  ///
  virtual bool isCaching() = 0;

  /// Is this variable is a member of a Structure?
  ///
  /// @return bool value
  ///
  virtual bool isMemberOfStructure() = 0;

  /// Is this variable metadata?
  ///
  /// @return true or false
  ///
  virtual bool isMetadata() = 0;

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

  /// Get the value as a unsigned char for a scalar DataItem. May also be
  /// one-dimensional of length 1.
  ///
  /// @return unsigned char object
  ///
  virtual unsigned char readScalarByte() throw ( Exception ) = 0;

  /// Get the value as a double for a scalar DataItem. May also be
  /// one-dimensional of length 1.
  ///
  /// @return double value
  ///
  virtual double readScalarDouble() throw ( Exception ) = 0;

  /// Get the value as a float for a scalar DataItem. May also be
  /// one-dimensional of length 1.
  ///
  /// @return float value
  ///
  virtual float readScalarFloat() throw ( Exception ) = 0;

  /// Get the value as a int for a scalar DataItem. May also be one-dimensional
  /// of length 1.
  ///
  /// @return integer value
  ///
  virtual int readScalarInt() throw ( Exception ) = 0;

  /// Get the value as a long for a scalar DataItem. May also be
  /// one-dimensional of length 1.
  ///
  /// @return long value
  ///
  virtual long readScalarLong() throw ( Exception ) = 0;

  /// Get the value as a short for a scalar DataItem. May also be
  /// one-dimensional of length 1.
  ///
  /// @return short value
  ///
  virtual short readScalarShort() throw ( Exception ) = 0;

  /// Get the value as a string. May also be one-dimensional of length 1.
  /// May also be one-dimensional of type CHAR,
  /// which will be turned into a single String.
  ///
  /// @return string object
  ///
  virtual std::string readString() throw ( Exception ) = 0;

  /// Remove an Attribute : uses the attribute hashCode to find it.
  ///
  /// @param attr IAttribute object
  /// @return true if was found and removed
  ///
  virtual bool removeAttribute(const IAttributePtr& attr) = 0;

  /// Set the data cache.
  ///
  /// @param cacheData
  ///           Array object
  /// @param isMetadata
  ///           : synthesised data, set true if must be saved in NcML output
  ///           (i.e. data not actually in the file).
  /// @throw  Exception
  ///            invalid type
  ///
  //## virtual void setCachedData(Array& cacheData, bool isMetadata) throw ( Exception ) = 0;

  /// Set whether to cache or not. Implies that the entire array will be
  /// stored, once read. Normally this is set automatically based on size of
  /// data.
  ///
  /// @param caching
  ///           set if caching.
  ///
  virtual void setCaching(bool caching) = 0;

  /// Set the data type.
  ///
  /// @param dataType
  ///           Class object
  ///
  virtual void setDataType(const std::type_info& dataType) = 0;

  /// Set the dimensions using the dimensions names. The dimension is searched
  /// for recursively in the parent groups.
  ///
  /// @param dimString : whitespace separated list of dimension names, or '*' for Dimension.UNKNOWN.
  ///
  virtual void setDimensions(const std::string& dimString) = 0;

  /// Set the dimension on the specified index.
  ///
  /// @param dim IDimension to add to this data item
  /// @param ind Index the dimension matches
  ///
  virtual void setDimension(const IDimensionPtr& dim, int ind) throw ( Exception ) = 0;

  /// Set the element size. Usually elementSize is determined by the dataType,
  /// use this only for exceptional cases.
  ///
  /// @param elementSize integer value
  ///
  virtual void setElementSize(int elementSize) = 0;

  /// Set sizeToCache.
  ///
  /// @param sizeToCache integer value
  ///
  virtual void setSizeToCache(int sizeToCache) = 0;

  /// Set the units of the DataItem.
  ///
  /// @param units string object Created on 20/03/2008
  ///
  virtual void setUnitsString(const std::string& units) = 0;

  /// Clone this data item. Return a new DataItem instance but share the same
  /// Array data storage.
  ///
  /// @return new DataItem instance
  ///
  virtual IDataItemPtr clone() = 0;
 };
 
 typedef std::list<IDataItemPtr> DataItemList;

} //namespace cdma

#endif //__CDMA_IDATAITEM_H__

