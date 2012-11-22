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

#ifndef __CDMA_FACTORY_H__
#define __CDMA_FACTORY_H__

#include <vector>
#include <string>
#include <map>
#include <typeinfo>

// CDMA
#include <cdma/Common.h>
#include <cdma/IDataSource.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/array/IArray.h>

namespace cdma
{

//==============================================================================
/// @brief Entry point for CDMA client
///
/// Client application get access to datasets from static methods of this class
//==============================================================================
class CDMA_DECL Factory 
{
public:

  /// Initialize the factory
  ///
  /// @param plugin_path Path to the plugins location
  ///
  static void init(const std::string &plugin_path);
  
  /// Clean up memory
  /// !!! Call this method before program exiting
  ///
  static void cleanup();

  static void setActiveView( const std::string& experiment);

  static const std::string& getActiveView();

  /// According to the currently defined experiment, this method will return the path
  /// to reach the declarative dictionary. It means the file where
  /// is defined what should be found in a IDataset that fits the experiment.
  /// It's a descriptive file.
  ///
  /// @return the path to the standard declarative file
  ///
  static std::string getKeyDictionaryPath();

  /// This method returns the path of the data definition documents.
  /// This path is not plug-in dependent.
  ///
  /// @return the path to the standard declarative file
  ///
  static std::string getKeyDictionaryFolder();

  /// According to the given factory this method will return the path to reach
  /// the folder containing mapping dictionaries. This file associate entry
  /// keys to paths that are plug-in dependent.
  ///
  /// @param factory_ptr Plug-in instance from which we want to load the dictionary
  /// @return the path to the plug-in's mapping dictionaries folder
  ///
  static std::string getMappingDictionaryFolder(const std::string& plugin_id);

  /// This method returns the path of the concepts dictionaries.
  /// This path is not plug-in dependent.
  ///
  /// @return the path to the concepts dictionaries folder
  ///
  static std::string getConceptDictionaryFolder();

  /// Get the folder path where to search for key dictionary files.
  /// This folder should contains all dictionaries that the above application needs.
  /// @return string path targeting a folder
  ///
  static std::string getDictionariesFolder();

  /// Retrieve the dataset referenced by the string.
  ///
  /// @param uri  yat::URI object of the destination
  /// @return     pair having first: IDataset and second: IPluginFactory
  /// @throw      Exception
  ///
  static IDatasetPtr openDataset(const std::string& uri) throw ( Exception );

  /// Open a dictionary document
  ///
  /// @param filepath Document file path
  /// @return     Shared pointer on a Dictionary object
  /// @throw      Exception
  ///
  static DictionaryPtr openDictionary(const std::string& filepath) throw ( Exception );

  /// According to a dataset location the factory will try to returns the datasource object
  /// that belong to the most suitable plugin
  ///
  /// @param uri Location (conform to the RFC 3986 specification) of a dataset
  /// @return IDataSourcePtr implementation of a IDataSource
  ///
  static IDataSourcePtr getDataSource(const std::string& uri) throw ( Exception );

  /// Create an View object with a given shape, a start position and a 
  ///
  /// @param shape    shape of the new array
  /// @return         Shared pointer on new View
  ///
  static IViewPtr createView(std::vector<int> shape, std::vector<int> start, std::vector<int> stride);

  /// Create an Key object with a given shape, a start position and a 
  ///
  /// @param name     keyword
  /// @param type     key type (@see IKey definition) 
  /// @return         Shared pointer on new key
  ///
  static IKeyPtr createKey(const std::string& name, IKey::Type type = IKey::UNDEFINED);

  /// Create an Array object with a given data type and shape
  ///
  /// @param type     data type
  /// @param shape    shape of the new array
  /// @return         Shared pointer on new Array
  ///
  static IArrayPtr createArray(const std::type_info& type, const std::vector<int> shape);

#if !defined(CDMA_NO_TEMPLATES)

  /// Create an Array with a given data type and shape
  ///
  /// @param shape    shape of the new array
  /// @return         Shared pointer on new Array
  ///
  /// @note Not yet implemented
  ///
  template<typename T> static IArrayPtr createArray(const std::vector<int> shape);

#endif // !CDMA_NO_TEMPLATES

private:
  // Only static methods, no instance
  Factory() {}
};

} //namespace CDMACore

#if !defined(CDMA_NO_TEMPLATES)
  #include <cdma/factory/impl/Factory.hpp>
#endif // !CDMA_NO_TEMPLATES

#endif //__CDMA_FACTORY_H__

