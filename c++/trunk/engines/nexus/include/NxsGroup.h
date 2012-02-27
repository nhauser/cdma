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

#include <list>
#include <map>
#include <string>

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IGroup.h>

#include <internal/common.h>
#include <NxsAttribute.h>
#include <NxsDataset.h>

namespace cdma
{
typedef std::map<yat::String, cdma::IGroupPtr> MapStringGroup;
typedef std::map<yat::String, cdma::IDataItemPtr> MapStringDataItem;
typedef std::map<yat::String, cdma::IAttributePtr> MapStringAttribute;

//==============================================================================
/// IGroup implementation for NeXus engine
/// See IGroup definition for more explanation
//==============================================================================
class NxsGroup : public cdma::IGroup
{
private:
  NxsDataset*           m_dataset_ptr;  // C-style pointer to the parent dataset
  NxsGroup*             m_root_ptr;     // TODO appeler celui du Dataset
  yat::String           m_path;         // Group path inside the Nexus File
  MapStringGroup        m_mapGroups;
  MapStringDataItem     m_mapDataItems;
  MapStringAttribute    m_attributes_map;
  bool                  m_bChildren;      // true if childs are enumerated
  bool                  m_attributes_loaded; // 'true' when the attributes are loaded from the data source

  void PrivEnumChildren();
  void PrivEnumAttributes();
  
public:

  //@{ Constructors

    NxsGroup(NxsDataset* dataset_ptr);
    NxsGroup(NxsDataset* dataset_ptr, const yat::String& parent_path, const yat::String& name);
    NxsGroup(NxsDataset* dataset_ptr, const yat::String& full_path);
    ~NxsGroup();

  //@} --------------------------------

  //@{ plug-in specific
  
    void setPath(const yat::String& strPath) { m_path = strPath; }
    std::string getPath() const;

  //@} --------------------------------
  
  //@{ IGroup interface
  
  bool isRoot() const;
  bool isEntry() const;
  cdma::IGroupPtr getRoot() const;
  cdma::IGroupPtr getParent() const;
  cdma::IDataItemPtr getDataItem(const std::string& short_name) throw ( cdma::Exception );
  cdma::IDataItemPtr getDataItemWithAttribute(const std::string& name, const std::string& value);
  cdma::IDimensionPtr getDimension(const std::string& name);
  cdma::IAttributePtr getAttribute(const std::string&);
  cdma::IGroupPtr getGroup(const std::string& short_name);
  cdma::IGroupPtr getGroupWithAttribute(const std::string& attributeName, const std::string& value);
  std::list<cdma::IAttributePtr> getAttributeList();
  std::list<cdma::IDataItemPtr> getDataItemList();
  std::list<cdma::IDimensionPtr> getDimensionList();
  std::list<cdma::IGroupPtr> getGroupList();
  cdma::IDataItemPtr addDataItem(const std::string& short_name);
  cdma::IDimensionPtr addDimension(const std::string& short_name);
  cdma::IGroupPtr addSubgroup(const std::string& short_name);
  bool removeDataItem(const cdma::IDataItemPtr& item);
  bool removeDataItem(const std::string& varName);
  bool removeDimension(const std::string& dimName);
  bool removeGroup(const cdma::IGroupPtr& group);
  bool removeGroup(const std::string& short_name);
  bool removeDimension(const cdma::IDimensionPtr& dimension);
  void setParent(const cdma::IGroupPtr&);
  cdma::IGroupPtr clone();

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
    cdma::IContainer::Type getContainerType() const { return cdma::IContainer::DATA_GROUP; }

  //@} --------------------------------
};

typedef yat::SharedPtr<NxsGroup, yat::Mutex> NxsGroupPtr;

}
#endif
