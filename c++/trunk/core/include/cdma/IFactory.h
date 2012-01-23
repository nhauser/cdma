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

#include <vector>
#include <string>
#include <typeinfo>

#include <yat/memory/SharedPtr.h>
#include <yat/plugin/IPlugInObject.h>

#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>

namespace cdma
{

//==============================================================================
/// Interface IFactory
//==============================================================================
class CDMA_DECL IFactory : public yat::IPlugInObject
{
public:
/*
  /// d-tor
  virtual ~IFactory()
  {
  }
*/
  /// Retrieve the dataset referenced by the path.
  /// 
  /// @param path can be either the string representation of an uri (see RFC 3986) or a file path
  /// @return CDMA Dataset
  /// @throw  Exception
  ///
  virtual IDatasetPtr openDataset(const std::string& path) throw ( Exception ) = 0;

  /// Retrieve the dataset referenced by an uri object.
  /// 
  /// @param uri string object
  /// @return CDMA Dataset
  /// @throw  Exception
  ///
  //virtual IDatasetPtr openDataset(const yat::URI& uri) throw ( Exception ) = 0;
  
  /// Open a dictionary
  /// 
  /// @param uri string object
  /// @return CDMA Dataset
  /// @throw  Exception
  ///
  virtual DictionaryPtr openDictionary(const std::string& filepath) throw ( Exception ) = 0;
  
  /// Returns the path of the mapping document for dictionary mechanism
  /// 
  /// @param dataset object
  /// @return path
  /// @throw  Exception
  ///
// virtual const std::string& getMappingFilePath(IDatasetPtr &dataset) throw ( Exception ) = 0;
  

  /// Create a CDMA Dataset with a string reference. If the file exists, it will
  ///
  /// @param uri string object
  /// @return CDMA Dataset
  /// @throw  Exception
  ///
  virtual IDatasetPtr createDatasetInstance(const std::string& uri) throw ( Exception ) = 0;

  /// Create a CDMA Dataset in memory only. The dataset is not open yet. It is
  /// necessary to call dataset.open() to access the root of the dataset.
  ///
  /// @return a CDMA Dataset
  /// @throw  Exception
  ///
  virtual IDatasetPtr createEmptyDatasetInstance() throw ( Exception ) = 0;

  virtual yat::SharedPtr<IPathParamResolver, yat::Mutex> createPathParamResolver( const PathPtr& ptrPath ) = 0;

  /// Return the symbol used by the plug-in to separate nodes in a string path
  /// @return
  /// @note <b>EXPERIMENTAL METHOD</b> do note use/implements
  ///
  virtual std::string getPathSeparator() = 0;

  /// The factory has a unique name that identifies it.
  /// @return the factory's name
  ///
  virtual std::string getName() = 0;

  /// Returns the URI detector of the instantiated plug-in. 
  ///
	/// @return the plugin's mimplementation of the IDataSource interface
	///
	virtual IDataSourcePtr getPluginURIDetector() = 0;
};

} //namespace CDMACore
#endif //__CDMA_IFACTORY_H__

