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
#ifndef __CDMA_NEXUS_DATAITEM_H__
#define __CDMA_NEXUS_DATAITEM_H__

#ifndef __TYPEINFO_INCLUDED__
 #include <typeinfo>
 #define __TYPEINFO_INCLUDED__
#endif

// CDMA Core
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/array/IArray.h>

// Plug-in
#include <internal/common.h>
#include <NxsAttribute.h>
#include <NxsGroup.h>

namespace cdma
{
namespace nexus
{

//==============================================================================
/// IDataItem implementation for NeXus engine
/// See IDataItem definition for more explanation
//==============================================================================
class CDMA_NEXUS_DECL DataItem : public IDataItem
{
public:
  typedef std::map<std::string, IAttributePtr> AttributeMap;
  typedef std::map<std::string, int>           DimOrderMap;

private:
  AttributeMap      m_attr_map;     // Attribute map: map[attr_name] = IAttributePtr
//  DimOrderMap       m_order_map;    // Dimension order map: map[dim_name] = order of the dimension
  
  Dataset*          m_dataset_ptr;  // C-style pointer in order to solve the circular reference
  yat::String       m_name;         // Name of the dataitem (ie: attribute long_name else node's name)
  yat::String       m_shortName;    // Short name of the node (ie: the key name when using the dictionary)
  yat::String       m_nodeName;     // physical name in NeXus file
  yat::String       m_path;         // Path of the item through the dataset file structure (excluding item node name)
  NexusDataSetInfo  m_item;         // Info on the belonged data
  IArrayPtr         m_array_ptr;    // Array object
  std::vector<int>  m_shape;        // Shape defined by the NexusDatasetInfo
  bool              m_bDimension;   // Does dimension order map has been initialized

public:

  //@{ Constructors & Destructor

    DataItem(Dataset* dataset_ptr, const std::string& path);
    DataItem(Dataset* dataset_ptr, const IGroupPtr& parent, const std::string& name );
    DataItem(Dataset* dataset_ptr, const NexusDataSetInfo& item, const std::string& path);
    ~DataItem();

  //@} --------------------------------

  //@{ IDataItem interface

    IAttributePtr findAttributeIgnoreCase(const std::string& name);
    int findDimensionView(const std::string& name);
    IGroupPtr getParent();
    IGroupPtr getRoot();
    IArrayPtr getData(std::vector<int> position = std::vector<int>() ) throw ( Exception );
    IArrayPtr getData(std::vector<int> origin, std::vector<int> shape) throw ( Exception );
    std::string getDescription();
    std::list<IDimensionPtr> getDimensions(int i);
    std::list<IDimensionPtr> getDimensionList();
    std::string getDimensionsString();
    int getElementSize();
    int getRank();
    std::vector<int> getShape();
    long getSize();
    IDataItemPtr getSlice(int dim, int value) throw ( Exception );
    const std::type_info& getType();
    std::string getUnit();
    bool isScalar();
    bool isUnlimited();
    bool isUnsigned();
    void setDataType(const std::type_info& dataType);
    void setData(const IArrayPtr&);
    void setDimension(const IDimensionPtr& dim, int ind) throw ( Exception );
    void setUnit(const std::string& units);
    IAttributePtr getAttribute(const std::string&);
    AttributeList getAttributeList();
    void setParent(const IGroupPtr&);
  
  //@} --------------------------------

  //@{ IContainer

    //cdma::IAttributePtr addAttribute(const std::string& short_name, yat::Any &value);
    void addAttribute(const IAttributePtr& attr);
    std::string getLocation() const;
    std::string getName() const;
    std::string getShortName() const;
    bool hasAttribute(const std::string&);
    bool removeAttribute(const IAttributePtr&);
    void setName(const std::string&);
    void setShortName(const std::string&);
    IContainer::Type getContainerType() const { return IContainer::DATA_ITEM; }

  //@} --------------------------------

  //@{plugin methods

    void setLocation(const std::string& path);
  
  //@} --------------------------------

protected:
  void init(Dataset* dataset_ptr, const std::string& path, bool init_from_file = true);

private:
  void loadArray();
  void checkArray();
  void initAttr();
  void open(bool openNode = true );
};

#ifdef CDMA_STD_SMART_PTR
typedef std::shared_ptr<DataItem> DataItemPtr;
#else
typedef yat::SharedPtr<DataItem, yat::Mutex> DataItemPtr;
#endif

} // namespace nexus
} // namespace cdma
#endif
