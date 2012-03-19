//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_NXSDATAITEM_H__
#define __CDMA_NXSDATAITEM_H__

#ifndef __TYPEINFO_INCLUDED__
 #include <typeinfo>
 #define __TYPEINFO_INCLUDED__
#endif

// Tools lib
#include <list>
#include <vector>
#include <yat/utils/String.h>

// CDMA Core
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/array/Array.h>

namespace cdma
{

//==============================================================================
/// This implementation is aimed to help plugin developper when special data 
/// access methods have to be implemented.
/// Such a method is invoked through the dictionary mechanism when a keyword
/// is not directly mapped whith a path to the data file organization, but
/// is mapped to this method through a <call> tag.
/// Use this implementation only when all data corresponding to the corresponding
/// keyword request may be easely contained in the computer RAM like scalars, spectrum,
/// images value but not stack of hundred or thousands images. For the laters cases
/// consider developping a specific implementation.
//==============================================================================
class CDMA_DECL SimpleDataItem : public IDataItem
{
private:
  std::list<IAttributePtr> m_attr_list;
  IDataset*                m_dataset_ptr;
  yat::String              m_name;        // Name of the dataitem
  ArrayPtr                 m_array_ptr;       // Array object

public:

  /// c-tor
  SimpleDataItem(IDataset* dataset_ptr, ArrayPtr ptrArray, const std::string &name);
  
  /// d-tor
  ~SimpleDataItem() { CDMA_FUNCTION_TRACE("SimpleDataItem::~SimpleDataItem"); };

  //@{ IDataItem interface

    IAttributePtr findAttributeIgnoreCase(const std::string& name);
    int findDimensionView(const std::string& name);
    IDataItemPtr getASlice(int dimension, int value) throw ( cdma::Exception );
    IGroupPtr getParent();
    IGroupPtr getRoot();
    ArrayPtr getData(std::vector<int> position = std::vector<int>() ) throw ( cdma::Exception );
    ArrayPtr getData(std::vector<int> origin, std::vector<int> shape) throw ( cdma::Exception );
    std::string getDescription();
    std::list<IDimensionPtr> getDimensions(int i);
    std::list<IDimensionPtr> getDimensionList();
    std::string getDimensionsString();
    int getElementSize();
    std::string getNameAndDimensions();
    std::string getNameAndDimensions(bool useFullName, bool showDimLength);
    std::list<RangePtr> getRangeList();
    int getRank();
    IDataItemPtr getSection(std::list<RangePtr> section) throw ( cdma::Exception );
    std::list<RangePtr> getSectionRanges();
    std::vector<int> getShape();
    long getSize();
    int getSizeToCache();
    IDataItemPtr getSlice(int dim, int value) throw ( cdma::Exception );
    const std::type_info& getType();
    std::string getUnitsString();
    bool hasCachedData();
    void invalidateCache();
    bool isCaching();
    bool isMemberOfStructure();
    bool isMetadata();
    bool isScalar();
    bool isUnlimited();
    bool isUnsigned();
    unsigned char readScalarByte() throw ( cdma::Exception );
    double readScalarDouble() throw ( cdma::Exception );
    float readScalarFloat() throw ( cdma::Exception );
    int readScalarInt() throw ( cdma::Exception );
    long readScalarLong() throw ( cdma::Exception );
    short readScalarShort() throw ( cdma::Exception );
    std::string readString() throw ( cdma::Exception );
    void setCaching(bool caching);
    void setDataType(const std::type_info& dataType);
    void setDimensions(const std::string& dimString);
    void setDimension(const IDimensionPtr& dim, int ind) throw ( cdma::Exception );
    void setElementSize(int elementSize);
    void setSizeToCache(int sizeToCache);
    void setUnitsString(const std::string& units);
    IDataItemPtr clone();
    IAttributePtr getAttribute(const std::string&);
    std::list<IAttributePtr > getAttributeList();
    void setParent(const IGroupPtr&);
    IDatasetPtr getDataset();
    void setData(const cdma::ArrayPtr& array);
  
  //@} --------------------------------

  //@{ IContainer

    //cdma::IAttributePtr addAttribute(const std::string& name, yat::Any &value);
    void addAttribute(const cdma::IAttributePtr& attr);
    std::string getLocation() const;
    std::string getName() const;
    std::string getShortName() const;
    bool hasAttribute(const std::string&);
    bool removeAttribute(const IAttributePtr&);
    void setName(const std::string&);
    void setShortName(const std::string&);
    cdma::IContainer::Type getContainerType() const { return cdma::IContainer::DATA_ITEM; }

  //@} --------------------------------

  //@{ plugin implementation should be call some of the following methods to properly initialize this object
  
  void setLogicalLocation(const std::string& location);
  
  //@} --------------------------------
};

//==============================================================================
/// This implementation is a specialization of the SimpleDataItem
/// intended to simplify the creation of DataItem object related to a scalar
/// value
//==============================================================================
class CDMA_DECL ScalarDataItem : public SimpleDataItem
{
  template <class T>
  ScalarDataItem(IDataset* dataset, T value, const std::string &name);
};

//==============================================================================
/// This implementation is a specialization of the SimpleDataItem
/// intended to simplify the creation of DataItem object related to a 1-d array
/// of values
//==============================================================================
class CDMA_DECL OneDimArrayDataItem : public SimpleDataItem
{
  template <class T>
  OneDimArrayDataItem(IDataset* dataset, T* values, yat::uint32 size, const std::string &name);
};

//==============================================================================
/// This implementation is a specialization of the SimpleDataItem
/// intended to simplify the creation of DataItem object related to a 2-d array
/// of values
//==============================================================================
class CDMA_DECL TwoDimArrayDataItem : public SimpleDataItem
{
  template <class T>
  TwoDimArrayDataItem(IDataset* dataset, T* values, yat::uint32 x_size, yat::uint32 y_size, const std::string &name);
};

}
#endif
