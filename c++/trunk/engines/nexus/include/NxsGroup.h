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

#include <cdma/exception/Exception.h>
#include <cdma/navigation/IGroup.h>

#include <internal/common.h>
#include <NxsDataset.h>

namespace cdma
{

class NxsGroup : public cdma::IGroup
{
private:
  NxsDataset*                    m_pDataset;       // Simple pointer to the parent dataset
//  NexusFilePtr                   m_ptrNxFile;
  IGroup*                        m_pParentGroup;   // Reference to the parent group
  IGroup*                        m_pRootGroup; // TODO appeler celui du Dataset
  yat::String                    m_strPath;        // Group path inside the Nexus File
  std::map<yat::String, cdma::IGroupPtr>     m_mapGroups;
  std::map<yat::String, cdma::IDataItemPtr>  m_mapDataItems;
  std::map<yat::String, cdma::IAttributePtr> m_mapAttributes;
  bool                           m_bChildren;      // true if childs are enumerated
  bool                           m_bAttributes;    // true if attributes are enumerated
  
  void PrivEnumChildren();
  void PrivEnumAttributes();

public:
	NxsGroup(NxsDataset* pDataset);
	NxsGroup(NxsDataset* pDataset, const yat::String& parent_path, const yat::String& name);
	NxsGroup(NxsDataset* pDataset, const yat::String& full_path);
	~NxsGroup();

  //@{ plug-in specific
  
//  void setFile(const NexusFilePtr& ptrFile) { m_ptrNxFile = ptrFile; }
  void setPath(const yat::String& strPath) { m_strPath = strPath; }
  
  //@}
  
  //@{ IGroup interface
  
	bool isRoot();
	bool isEntry();
	std::string getLocation();
  std::string getPath();
  std::string getName();
  std::string getShortName();
	cdma::IGroupPtr getParent();
	cdma::IGroupPtr getRoot();
	cdma::IDataItemPtr getDataItem(const std::string& shortName) throw ( cdma::Exception );
  cdma::IDataItemPtr getDataItemWithAttribute(const std::string& name, const std::string& value);
  cdma::IDimensionPtr getDimension(const std::string& name);
	cdma::IAttributePtr getAttribute(const std::string&);
  cdma::IGroupPtr getGroup(const std::string& shortName);
  cdma::IGroupPtr getGroupWithAttribute(const std::string& attributeName, const std::string& value);
	bool hasAttribute(const std::string& name, const std::string& value);
	std::list<cdma::IAttributePtr> getAttributeList();
	std::list<cdma::IDataItemPtr> getDataItemList();
	std::list<cdma::IDimensionPtr> getDimensionList();
	std::list<cdma::IGroupPtr> getGroupList();
	void addDataItem(const cdma::IDataItemPtr& v);
	void addOneDimension(const cdma::IDimensionPtr& dimension)  throw ( cdma::Exception );
	void addSubgroup(const cdma::IGroupPtr& group) throw ( cdma::Exception );
	void addOneAttribute(const cdma::IAttributePtr&);
	void addStringAttribute(const std::string&, const std::string&);
	bool removeDataItem(const cdma::IDataItemPtr& item);
	bool removeDataItem(const std::string& varName);
	bool removeDimension(const std::string& dimName);
	bool removeGroup(const cdma::IGroupPtr& group);
	bool removeGroup(const std::string& shortName);
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
