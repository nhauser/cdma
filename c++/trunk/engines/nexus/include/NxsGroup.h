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

#include <cdma/IObject.h>
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
  NxsDatasetWPtr        m_dataset_wptr;  // Weak pointer to the parent dataset
  NexusFilePtr          m_ptrNxFile;     // Shared pointer to the nexus file
  mutable NxsGroupWPtr  m_parent_wptr;   // Reference to the parent group
  NxsGroupWPtr          m_root_wptr;     // TODO appeler celui du Dataset
  NxsGroupWPtr          m_self_wptr;     // self reference given to its childrens
  yat::String           m_strPath;       // Group path inside the Nexus File
  MapStringGroup        m_mapGroups;
  MapStringDataItem     m_mapDataItems;
  MapStringAttribute    m_attributes_map;
  bool                  m_bChildren;      // true if childs are enumerated
  bool                  m_attributes_loaded; // 'true' when the attributes are loaded from the data source

  void PrivEnumChildren();
  void PrivEnumAttributes();
  
public:
  //@{ Constructors

  NxsGroup(NxsDatasetWPtr dataset_wptr);
  NxsGroup(NxsDatasetWPtr dataset_wptr, const yat::String& parent_path, const yat::String& name);
  NxsGroup(NxsDatasetWPtr dataset_wptr, const yat::String& full_path);
  ~NxsGroup();

  //@}

  //@{ plug-in specific
  
  void setFile(const NexusFilePtr& ptrFile) { m_ptrNxFile = ptrFile; }
  void setPath(const yat::String& strPath) { m_strPath = strPath; }
  void setSelfRef(const NxsGroupPtr& ptr);
  std::string getPath() const;

  //@}
  
  //@{ IGroup interface
  
  bool isRoot() const;
  bool isEntry() const;
  cdma::IGroupPtr getRoot() const;
  std::string getLocation() const;
  std::string getName() const;
  std::string getShortName() const;
  cdma::IGroupPtr getParent() const;
  cdma::IDataItemPtr getDataItem(const std::string& short_name) throw ( cdma::Exception );
  cdma::IDataItemPtr getDataItemWithAttribute(const std::string& name, const std::string& value);
  cdma::IDimensionPtr getDimension(const std::string& name);
  cdma::IAttributePtr getAttribute(const std::string&);
  cdma::IGroupPtr getGroup(const std::string& short_name);
  cdma::IGroupPtr getGroupWithAttribute(const std::string& attributeName, const std::string& value);
  bool hasAttribute(const std::string&);
  std::list<cdma::IAttributePtr> getAttributeList();
  std::list<cdma::IDataItemPtr> getDataItemList();
  std::list<cdma::IDimensionPtr> getDimensionList();
  std::list<cdma::IGroupPtr> getGroupList();
  cdma::IDataItemPtr addDataItem(const std::string& short_name);
  cdma::IDimensionPtr addDimension(const std::string& short_name);
  cdma::IGroupPtr addSubgroup(const std::string& short_name);
  cdma::IAttributePtr addAttribute(const std::string& short_name, yat::Any &value);
  bool removeDataItem(const cdma::IDataItemPtr& item);
  bool removeDataItem(const std::string& varName);
  bool removeDimension(const std::string& dimName);
  bool removeGroup(const cdma::IGroupPtr& group);
  bool removeGroup(const std::string& short_name);
  bool removeDimension(const cdma::IDimensionPtr& dimension);
  bool removeAttribute(const cdma::IAttributePtr&);
  void setName(const std::string&);
  void setShortName(const std::string&);
  void setParent(const cdma::IGroupPtr&);
  cdma::IGroupPtr clone();
  //@} IGroup interface
  
  //@{IObject interface
  cdma::CDMAType::ModelType getModelType() const { return cdma::CDMAType::Group; };
  std::string getFactoryName() const { return NXS_FACTORY_NAME; };
  //@} IObject interface

};

typedef yat::SharedPtr<NxsGroup, yat::Mutex> NxsGroupPtr;

}
#endif