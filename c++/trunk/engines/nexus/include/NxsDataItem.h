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
class CDMA_NEXUS_DECL NxsDataItem : public IDataItem
{
public:
  typedef std::map<std::string, IAttributePtr> AttributeMap;
  typedef std::map<std::string, int>           DimOrderMap;

private:
  AttributeMap      m_attr_map;     // Attribute map: map[attr_name] = IAttributePtr
//  DimOrderMap       m_order_map;    // Dimension order map: map[dim_name] = order of the dimension
  
  NxsDataset*       m_dataset_ptr;  // C-style pointer in order to solve the circular reference
  yat::String       m_name;         // Name of the dataitem (ie: attribute long_name else node's name)
  yat::String       m_shortName;    // Short name of the node (physical name in NeXus file)
  yat::String       m_path;         // Path of the item through the dataset file structure (excluding item node name)
  NexusDataSetInfo  m_item;         // Info on the belonged data
  ArrayPtr          m_array_ptr;    // Array object
  std::vector<int>  m_shape;        // Shape defined by the NexusDatasetInfo
  bool              m_bDimension;   // Does dimension order map has been initialized

public:

  //@{ Constructors & Destructor

    NxsDataItem(NxsDataset* dataset_ptr, const std::string& path);
    NxsDataItem(NxsDataset* dataset_ptr, const IGroupPtr& parent, const std::string& name );
    NxsDataItem(NxsDataset* dataset_ptr, const NexusDataSetInfo& item, const std::string& path);
    ~NxsDataItem();

  //@} --------------------------------

  //@{ IDataItem interface

    IAttributePtr findAttributeIgnoreCase(const std::string& name);
    int findDimensionView(const std::string& name);
    IGroupPtr getParent();
    IGroupPtr getRoot();
    ArrayPtr getData(std::vector<int> position = std::vector<int>() ) throw ( cdma::Exception );
    ArrayPtr getData(std::vector<int> origin, std::vector<int> shape) throw ( cdma::Exception );
    std::string getDescription();
    std::list<IDimensionPtr> getDimensions(int i);
    std::list<IDimensionPtr> getDimensionList();
    std::string getDimensionsString();
    int getElementSize();
    int getRank();
    std::vector<int> getShape();
    long getSize();
    IDataItemPtr getSlice(int dim, int value) throw ( cdma::Exception );
    const std::type_info& getType();
    std::string getUnitsString();
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
    void setDataType(const std::type_info& dataType);
    void setData(const cdma::ArrayPtr&);
    void setDimensions(const std::string& dimString);
    void setDimension(const IDimensionPtr& dim, int ind) throw ( cdma::Exception );
    void setUnitsString(const std::string& units);
    IAttributePtr getAttribute(const std::string&);
    AttributeList getAttributeList();
    void setParent(const IGroupPtr&);
    IDatasetPtr getDataset();
  
  //@} --------------------------------

  //@{ IContainer

    //cdma::IAttributePtr addAttribute(const std::string& short_name, yat::Any &value);
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

  //@{plugin methods

    void setLocation(const std::string& path);
  
  //@} --------------------------------

protected:
  void init(NxsDataset* dataset_ptr, const std::string& path, bool init_from_file = true);

private:
  void loadArray();
  void checkArray();
  void initAttr();
  void open(bool openNode = true );
};

typedef yat::SharedPtr<NxsDataItem, yat::Mutex> NxsDataItemPtr;
}
#endif
