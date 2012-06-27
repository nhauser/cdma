//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IFACTORY_H__
#define __CDMA_IFACTORY_H__

#include <yat/plugin/IPlugInObject.h>

#include <cdma/Common.h>
#include <cdma/IDataSource.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/dictionary/Dictionary.h>

namespace cdma
{

/// @cond pluginAPI

//==============================================================================
/// @brief Abstraction of the factory object that each plug-ins must provide
//==============================================================================
class CDMA_DECL IFactory : public yat::IPlugInObject
{
public:
  /// d-tor
  virtual ~IFactory()
  {
    CDMA_FUNCTION_TRACE("IFactory::~IFactory");
  }

  /// Retrieve the dataset referenced by the path.
  /// 
  /// @param dataset_location string representation of an uri (see RFC 3986) locating the dataset
  ///
  /// @return shared pointer to a IDataset instance
  ///
  /// @throw  Exception
  ///
  virtual IDatasetPtr openDataset(const std::string& dataset_location) throw ( Exception ) = 0;
  
  /// Open a dictionary
  /// 
  /// @param file_path file path to the dictionary document
  ///
  /// @return shared pointer to a dictionary object
  ///
  /// @throw  Exception
  ///
  virtual DictionaryPtr openDictionary(const std::string& file_path) throw ( Exception ) = 0;
  
  /// Create a CDMA Dataset with a string reference. If the file exists, it will
  ///
  /// @param uri string object
  ///
  /// @return shared pointer to a IDataset instance
  ///
  /// @throw  Exception
  ///
  virtual IDatasetPtr createDatasetInstance(const std::string& uri) throw ( Exception ) = 0;

  /// Create a CDMA Dataset in memory only. The dataset is not open yet. It is
  /// necessary to call dataset.open() to access the root of the dataset.
  ///
  /// @return shared pointer to a IDataset instance
  ///
  /// @throw  Exception
  ///
  virtual IDatasetPtr createEmptyDatasetInstance() throw ( Exception ) = 0;

  /// Return the symbol used by the plug-in to separate nodes in a string path
  ///
  /// @note <b>EXPERIMENTAL METHOD</b> do note use/implements
  ///
  virtual std::string getPathSeparator() = 0;

  /// The factory has a unique name that identifies it.
  ///
  /// @return the factory's name
  ///
  virtual std::string getName() = 0;

  /// Returns the URI detector of the instantiated plug-in. 
  ///
  /// @return the plugin's mimplementation of the IDataSource interface
  ///
  virtual IDataSourcePtr getPluginURIDetector() = 0;

  /// Returns the plugin methods list
  virtual std::list<std::string> getPluginMethodsList() = 0;
};

DECLARE_SHARED_PTR(IFactory);

/// @endcond pluginAPI

} //namespace CDMACore
#endif //__CDMA_IFACTORY_H__

