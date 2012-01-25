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

// #include <yat/utils/URI.h>
#include <internal/common.h>
#include <cdma/IObject.h>
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
  yat::String                           m_uri;          ///< uniform resource indentifier to the dataset
  bool                                  m_open;         ///< is the data source opened
  NexusFilePtr                          m_ptrNxFile;    ///< handle on file
  IGroupPtr                             m_phy_root;     ///< document physical root
  LogicalGroupPtr                       m_log_root;     ///< document logical root
  std::map<yat::String, IGroupPtr>      m_group_map;    ///< association between groups and paths
  std::map<yat::String, IDataItemPtr>   m_item_map;     ///< association between data items and paths
  NxsDatasetWPtr                        m_self_wptr;    ///< weak self reference

public:
  /// Constructor
  ///  param :
  /// @param : filepath string representing the file path
  ///
  NxsDataset(const std::string& filepath);
  
  /// Constructor
  ///  param :
  /// @param : uri identifier to dataset location
  ///
  // NxsDataset(const yat::URI& uri);
  
  virtual ~NxsDataset() { }

  /// Set a reference on the object himself
  void setSelfRef(const NxsDatasetPtr& ptr);

  //@{ Plug-in methods
  // Path concatenation
  static yat::String concatPath(const yat::String &path, const yat::String& name);
  
  //@}
  
  //@{ IDataset interface
  const NexusFilePtr& getHandle() { return m_ptrNxFile; };
  
  /// Return the root group of the dataset.
  ///
  /// @return IGroup type Created on 16/06/2008
  ///
  IGroupPtr getRootGroup();

  /// Return the the logical root of the dataset.
  ///
  /// @return IGroup type Created on 16/06/2008
  ///
  LogicalGroupPtr getLogicalRoot();

  /// Return the location of the dataset. If it's a file, return the path.
  ///
  /// @return string type Created on 16/06/2008
  ///
  std::string getLocation();

  /// Return the title of the dataset.
  ///
  /// @return string type Created on 16/06/2008
  ///
  std::string getTitle();

  /// Set the location field of the dataset.
  ///
  /// @param location in string type
  ///
  void setLocation(const std::string& location);

  /// Set the title for the Dataset.
  ///
  /// @param title a string object
  ///
  void setTitle(const std::string& title);

  /// Synchronize the dataset with the file reference.
  ///
  /// @return true or false
  ///
  bool sync() throw ( cdma::Exception );

  /// Save the contents / changes of the dataset to the file.
  ///
  void save() throw ( cdma::Exception );

  /// Save the contents of the dataset to a new file.
  ///
  void saveTo(const std::string& location) throw ( cdma::Exception );

  /// Save the specific contents / changes of the dataset to the file.
  ///
  void save(const IContainer& container) throw ( cdma::Exception );

  /// Save the attribute to the specific path of the file.
  ///
  void save(const std::string& parentPath, const IAttributePtr& attribute) throw ( cdma::Exception );

  IGroupPtr    getGroupFromPath(const std::string &fullPath);
  IDataItemPtr getItemFromPath(const std::string &fullPath);
  IDataItemPtr getItemFromPath(const yat::String &path, const yat::String& name);

  //@} IDataset interface
  
  //@{IObject interface

  CDMAType::ModelType getModelType() const { return CDMAType::Dataset; };
  std::string getFactoryName() const { return NXS_FACTORY_NAME; };

  //@} IObject interface

};

typedef yat::SharedPtr<NxsDataset, yat::Mutex> NxsDatasetPtr;

//==============================================================================
/// Convenient class
//==============================================================================
class NexusFileAccess
{
private:
  NexusFilePtr m_ptrNxFile;
public:
  NexusFileAccess( const NexusFilePtr& ptrFile );
  ~NexusFileAccess();
};

} // namespace
#endif

