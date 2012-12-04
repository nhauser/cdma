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
#ifndef __CDMA_NEXUS_DATASET_H__
#define __CDMA_NEXUS_DATASET_H__

#include <yat/utils/String.h>
#include <yat/utils/URI.h>

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/factory/plugin/IPluginFactory.h>
#include <internal/common.h>

#include "nxfile.h"

namespace cdma
{
namespace nexus
{

//==============================================================================
/// IDataset implementation for NeXus engine
/// See IDataset definition for more explanation
//==============================================================================
class CDMA_NEXUS_DECL Dataset : public IDataset
{
public:
  typedef std::map<std::string, IContainerPtrList> ContainerCache;
  typedef std::map<std::string, IGroupPtr> GroupCache;
  typedef std::map<std::string, IDataItemPtr> DataItemCache;

protected:

  typedef std::map<std::string, NexusItemInfoList> NexusItemInfoListCache;

  yat::URI               m_location;      ///< uniform resource indentifier to the dataset
  IPluginFactory*        m_factory_ptr;   ///< C-style pointer on factory
  bool                   m_open;          ///< is the data source opened
  NexusFilePtr           m_file_handle;   ///< handle on the NeXus file
  IGroupPtr              m_phy_root;      ///< document physical root
  ILogicalGroupPtr       m_log_root;      ///< document logical root
  mutable GroupCache             m_group_map;     ///< association between groups and paths
  mutable DataItemCache          m_item_map;      ///< association between data items and paths
  mutable ContainerCache         m_container_map; ///< association between container list (dataitems or group) and paths
  mutable NexusItemInfoListCache m_node_cache;    ///< cache for method findAllContainerByPath

  // close dataset handle
  void close();

  /// Default constructor
  Dataset();

  /// Constructor
  ///
  /// @param location dataset location in URI form
  /// @param factory_ptr C-style pointer on the plugin factory creating this object
  ///
  Dataset( const yat::URI& location, IPluginFactory *factory_ptr );

public:

  //@{ engine methods -----------------

  /// Path concatenation
  static yat::String concatPath(const yat::String &path, const yat::String& name);
  
  /// Accessor on the NeXus file object
  const NexusFilePtr& getHandle() { return m_file_handle; };

  /// Returns a C-style pointer on the plugin factory who created this dataset
  IPluginFactory* getPluginFactory() const { return m_factory_ptr; };

  //@}
  
  //@{ IDataset interface -------------
  
  virtual ~Dataset();
  IGroupPtr getRootGroup();
  ILogicalGroupPtr getLogicalRoot();
  std::string getLocation();
  std::string getTitle();
  void setLocation(const std::string& location);
  void setLocation(const yat::URI& location);
  void setTitle(const std::string& title);
  bool sync() throw ( Exception );
  void save() throw ( Exception );
  void saveTo(const std::string& location) throw ( Exception );
  void save(const IContainer& container) throw ( Exception );
  void save(const std::string& parentPath, const IAttributePtr& attribute) throw ( Exception );
  IGroupPtr    getGroupFromPath(const std::string &fullPath);
  IDataItemPtr getItemFromPath(const std::string &fullPath);
  IDataItemPtr getItemFromPath(const yat::String &path, const yat::String& name);
  IContainerPtr findContainerByPath(const std::string& path);
  IContainerPtrList findAllContainerByPath(const std::string& path, bool first_only=false);

  //@} IDataset interface

private:
  void privExtractNextPathPart(yat::String* path_p, yat::String* name_p, yat::String* class_p);
  NexusItemInfoList privMatchingNodes(const yat::String& current_path, 
                                      const yat::String& name_pattern, 
                                      const yat::String& class_name);
  IContainerPtrList privFindContainer(const std::string& input_path, bool first_only);
};

#ifdef CDMA_STD_SMART_PTR
typedef std::shared_ptr<Dataset> DatasetPtr;
#else
typedef yat::SharedPtr<Dataset, yat::Mutex> DatasetPtr;
#endif

//==============================================================================
/// Convenient class
//==============================================================================
class CDMA_NEXUS_DECL NexusFileAccess
{
private:
  NexusFilePtr m_file_handle;
public:
  NexusFileAccess( const NexusFilePtr& handle );
  ~NexusFileAccess();
};

} // namespace nexus
} // namespace cdma
#endif

