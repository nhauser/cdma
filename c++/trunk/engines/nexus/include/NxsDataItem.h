// ****************************************************************************
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
// ****************************************************************************
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
#include <yat/memory/SharedPtr.h>

// CDMA Core
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/array/Array.h>

// Plug-in
#include <internal/common.h>
#include <NxsAttribute.h>
#include <NxsGroup.h>

namespace cdma
{

//==============================================================================
/// IDataItem implementation for NeXus engine
/// See IDataItem definition for more explanation
//==============================================================================
class CDMA_DECL NxsDataItem : public IDataItem
{
private:
  std::list<IAttributePtr> m_attr_list;
  NxsDatasetWPtr           m_dataset_wptr; // use a weakptr in order to solve the circular reference
  yat::String              m_name;         // Name of the dataitem (ie: attribute long_name else node's name)
  yat::String              m_shortName;
  yat::String              m_path;         // Path of the item through the dataset file structure (excluding item node name)
  NexusDataSetInfo         m_item;         // Info on the belonged data
  ArrayPtr                 m_array_ptr;    // Array object
  std::vector<int>         m_shape;        // Shape defined by the NexusDatasetInfo

public:

  //@{ Constructors & Destructor

    NxsDataItem(NxsDatasetWPtr dataset_wptr, const std::string& path, bool init_from_file = true );
    NxsDataItem(NxsDatasetWPtr dataset_wptr, const IGroupPtr& parent, const std::string& name );
    NxsDataItem(NxsDatasetWPtr dataset_wptr, const NexusDataSetInfo& item, const std::string& path);
    ~NxsDataItem();

  //@} --------------------------------

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
  
  //@} --------------------------------

  //@{ IContainer

    cdma::IAttributePtr addAttribute(const std::string& short_name, yat::Any &value);
    std::string getLocation() const;
    std::string getName() const;
    std::string getShortName() const;
    bool hasAttribute(const std::string&);
    bool removeAttribute(const IAttributePtr&);
    void setName(const std::string&);
    void setShortName(const std::string&);

  //@} --------------------------------

  //@{IObject interface

    CDMAType::ModelType getModelType() const { return CDMAType::DataItem; };
    std::string getFactoryName() const { return NXS_FACTORY_NAME; };

  //@} --------------------------------

  //@{plugin methods

    void setLocation(const std::string& path);
  
  //@} --------------------------------

protected:
  void init(NxsDatasetWPtr dataset_wptr, const std::string& path, bool init_from_file = true);
private:
  void loadArray();
  void checkArray();
  void initAttr();
  void open(bool openNode = true );
};

typedef yat::SharedPtr<NxsDataItem, yat::Mutex> NxsDataItemPtr;
}
#endif
