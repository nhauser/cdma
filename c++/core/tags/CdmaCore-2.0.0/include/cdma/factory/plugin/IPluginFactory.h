//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
//
// This file is part of cdma-core library.
//
// The cdma-core library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
//
// The CDMA library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
//
// Contributors :
// See AUTHORS file 
//******************************************************************************

#ifndef __CDMA_IPLUGINFACTORY_H__
#define __CDMA_IPLUGINFACTORY_H__

#include <yat/plugin/IPlugInObject.h>

#include <cdma/Common.h>
#include <cdma/IDataSource.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/dictionary/plugin/Dictionary.h>

namespace cdma
{

/// @cond pluginAPI

//==============================================================================
/// @brief Abstraction of the factory object that each plug-ins must provide
//==============================================================================
class CDMA_DECL IPluginFactory : public yat::IPlugInObject
{
public:
  /// d-tor
  virtual ~IPluginFactory()
  {
    CDMA_FUNCTION_TRACE("IPluginFactory::~IPluginFactory");
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

DECLARE_SHARED_PTR(IPluginFactory);

/// @endcond pluginAPI

} //namespace CDMACore

#endif //__CDMA_IPLUGINFACTORY_H__

