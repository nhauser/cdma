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
#ifndef __CDMA_NXSDATASET_H__
#define __CDMA_NXSDATASET_H__

#include <internal/common.h>
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/IFactory.h>

#include "nxfile.h"

namespace cdma
{

//==============================================================================
/// IDataset implementation for NeXus engine
/// See IDataset definition for more explanation
//==============================================================================
class CDMA_NEXUS_DECL NxsDataset : public IDataset
{
protected:
  yat::URI                              m_location;     ///< uniform resource indentifier to the dataset
  bool                                  m_open;         ///< is the data source opened
  NexusFilePtr                          m_file_handle;  ///< handle on the NeXus file
  IGroupPtr                             m_phy_root;     ///< document physical root
  LogicalGroupPtr                       m_log_root;     ///< document logical root
  std::map<yat::String, IGroupPtr>      m_group_map;    ///< association between groups and paths
  std::map<yat::String, IDataItemPtr>   m_item_map;     ///< association between data items and paths
  IFactory*                             m_factory_ptr;  ///< C-style pointer on factory

  // close dataset handle
  void close();

  /// Default constructor
  NxsDataset();

  /// Constructor
  ///
  /// @param location dataset location in URI form
  /// @param factory_ptr C-style pointer on the plugin factory creating this object
  ///
  NxsDataset( const yat::URI& location, IFactory *factory_ptr );

public:

  //@{ engine methods -----------------

  /// Path concatenation
  static yat::String concatPath(const yat::String &path, const yat::String& name);
  
  /// Accessor on the NeXus file object
  const NexusFilePtr& getHandle() { return m_file_handle; };

  /// Returns a C-style pointer on the plugin factory who created this dataset
  cdma::IFactory* getPluginFactory() const { return m_factory_ptr; };

  //@}
  
  //@{ IDataset interface -------------
  
  virtual ~NxsDataset();
  IGroupPtr getRootGroup();
  LogicalGroupPtr getLogicalRoot();
  std::string getLocation();
  std::string getTitle();
  void setLocation(const std::string& location);
  void setLocation(const yat::URI& location);
  void setTitle(const std::string& title);
  bool sync() throw ( cdma::Exception );
  void save() throw ( cdma::Exception );
  void saveTo(const std::string& location) throw ( cdma::Exception );
  void save(const IContainer& container) throw ( cdma::Exception );
  void save(const std::string& parentPath, const IAttributePtr& attribute) throw ( cdma::Exception );
  IGroupPtr    getGroupFromPath(const std::string &fullPath);
  IDataItemPtr getItemFromPath(const std::string &fullPath);
  IDataItemPtr getItemFromPath(const yat::String &path, const yat::String& name);

  //@} IDataset interface
};

typedef yat::SharedPtr<NxsDataset, yat::Mutex> NxsDatasetPtr;

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

} // namespace
#endif

