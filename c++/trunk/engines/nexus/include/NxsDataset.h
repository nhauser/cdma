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

#include "nxfile.h"

namespace cdma
{

//==============================================================================
/// IDataset implementation for NeXus engine
/// See IDataset definition for more explanation
//==============================================================================
class NxsDataset : public IDataset
{
protected:
  yat::URI                              m_location;     ///< uniform resource indentifier to the dataset
  bool                                  m_open;         ///< is the data source opened
  NexusFilePtr                          m_file_handle;  ///< handle on the NeXus file
  IGroupPtr                             m_phy_root;     ///< document physical root
  LogicalGroupPtr                       m_log_root;     ///< document logical root
  std::map<yat::String, IGroupPtr>      m_group_map;    ///< association between groups and paths
  std::map<yat::String, IDataItemPtr>   m_item_map;     ///< association between data items and paths
  NxsDatasetWPtr                        m_self_wptr;    ///< weak self reference

  // close dataset handle
  void close();

  /// Constructor
  ///
  /// @param : filepath string representing the file path
  ///
  NxsDataset(const yat::URI& location);

  /// Constructor
  ///
  NxsDataset();

public:
  
  //@{ engine methods -----------------

  /// Set a reference on the object himself
  void setSelfRef(const NxsDatasetPtr& ptr);

  // Path concatenation
  static yat::String concatPath(const yat::String &path, const yat::String& name);
  
  const NexusFilePtr& getHandle() { return m_file_handle; };

  //@}
  
  //@{ IDataset interface -------------
  
  virtual ~NxsDataset() { }
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
class NexusFileAccess
{
private:
  NexusFilePtr m_file_handle;
public:
  NexusFileAccess( const NexusFilePtr& handle );
  ~NexusFileAccess();
};

} // namespace
#endif

