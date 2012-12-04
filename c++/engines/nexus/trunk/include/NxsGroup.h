//*****************************************************************************
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
//*****************************************************************************
#ifndef __CDMA_NXSGROUP_H__
#define __CDMA_NXSGROUP_H__

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IGroup.h>

#include <internal/common.h>
#include <NxsAttribute.h>
#include <NxsDataset.h>

namespace cdma
{
namespace nexus
{

typedef std::map<yat::String, IGroupPtr> MapStringGroup;
typedef std::map<yat::String, IDataItemPtr> MapStringDataItem;
typedef std::map<yat::String, IAttributePtr> MapStringAttribute;
typedef std::map<std::string, IDimensionPtr> MapStringDimension;

//==============================================================================
/// IGroup implementation for NeXus engine
/// See IGroup definition for more explanation
//==============================================================================
class CDMA_NEXUS_DECL Group : public IGroup
{
private:
  Dataset*            m_dataset_ptr;       // C-style pointer to the parent dataset
  yat::String         m_path;              // Group path inside the Nexus File
  bool                m_bChildren;         // true if childs are enumerated
  bool                m_attributes_loaded; // 'true' when the attributes are loaded from the data source

  MapStringGroup      m_mapGroups;         // Group map: map[group_name]= group child
  MapStringDataItem   m_mapDataItems;      // DataItem map : map[name]=attribute
  MapStringDimension  m_mapDimensions;     // Dimension map: map[dim_name] = dimension
  MapStringAttribute  m_attributes_map;    // Attribute map : map[name]=attribute

  void PrivEnumChildren();
  void PrivEnumAttributes();
  
public:

  //@{ Constructors

    Group(Dataset* dataset_ptr);
    Group(Dataset* dataset_ptr, const yat::String& parent_path, const yat::String& name);
    Group(Dataset* dataset_ptr, const yat::String& full_path);
    ~Group();

  //@} --------------------------------

  //@{ plug-in specific
  
    void setPath(const yat::String& strPath) { m_path = strPath; }
    std::string getPath() const;

  //@} --------------------------------
  
  //@{ IGroup interface
  
  bool isRoot() const;
  bool isEntry() const;
  IGroupPtr getRoot() const;
  IGroupPtr getParent() const;
  IDataItemPtr getDataItem(const std::string& short_name) throw ( Exception );
  IDataItemPtr getDataItemWithAttribute(const std::string& name, const std::string& value);
  IDimensionPtr getDimension(const std::string& name);
  IAttributePtr getAttribute(const std::string&);
  IGroupPtr getGroup(const std::string& short_name);
  IGroupPtr getGroupWithAttribute(const std::string& attributeName, const std::string& value);
  std::list<IAttributePtr> getAttributeList();
  std::list<IDataItemPtr> getDataItemList();
  std::list<IDimensionPtr> getDimensionList();
  std::list<IGroupPtr> getGroupList();
  IDataItemPtr addDataItem(const std::string& short_name);
  IDimensionPtr addDimension(const IDimensionPtr& dim);
  IGroupPtr addSubgroup(const std::string& short_name);
  bool removeDataItem(const IDataItemPtr& item);
  bool removeDataItem(const std::string& varName);
  bool removeDimension(const std::string& dimName);
  bool removeGroup(const IGroupPtr& group);
  bool removeGroup(const std::string& short_name);
  bool removeDimension(const IDimensionPtr& dimension);
  void setParent(const IGroupPtr&);

  //@} --------------------------------
  
  //@{ IContainer

  //  IAttributePtr addAttribute(const std::string& short_name, yat::Any &value);
    void addAttribute(const IAttributePtr& attr);
    std::string getLocation() const;
    std::string getName() const;
    std::string getShortName() const;
    bool hasAttribute(const std::string&);
    bool removeAttribute(const IAttributePtr&);
    void setName(const std::string&);
    void setShortName(const std::string&);
    IContainer::Type getContainerType() const { return IContainer::DATA_GROUP; }

  //@} --------------------------------
};

#ifdef CDMA_STD_SMART_PTR
typedef std::shared_ptr<Group> GroupPtr;
#else
typedef yat::SharedPtr<Group, yat::Mutex> GroupPtr;
#endif


} // namespace nexus
} // namespace cdma

#endif
