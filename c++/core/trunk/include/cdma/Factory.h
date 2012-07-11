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

// yat
#include <yat/system/SysUtils.h>
#include <yat/plugin/PlugInManager.h>
#include <yat/plugin/IPlugInInfo.h>
#include <yat/utils/URI.h>

// CDMA
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/IFactory.h>
#include <cdma/dictionary/PluginMethods.h>
#include <cdma/array/Array.h>

namespace cdma
{

/// @internal For internal purpose
typedef cdma::IPluginMethod* (*GetMethodObject_t) ( void );

/// @cond internal

//==============================================================================
/// @brief For internal purpose
/// @internal
/// Specialization of the yat::PluginManager class in order to able to get access
/// to the plugins methods
//==============================================================================
class CDMA_DECL PlugInManager: public yat::PlugInManager { };

/// @endcond

//==============================================================================
/// @brief Entry point for CDMA client
///
/// Client application get access to datasets from static methods of this class
//==============================================================================
class CDMA_DECL Factory 
{
private:
  typedef std::map<std::string, IPluginMethodPtr> PluginMethodsMap;
  struct Plugin 
  {
    yat::IPlugInInfo*      info;
    yat::IPlugInFactory*   factory;
    yat::PlugIn*           plugin_objet;
    PluginMethodsMap       plugin_method_map;
  };
  
public:
  typedef std::map<std::string, Plugin> PluginMap;
  typedef std::map<std::string, IFactoryPtr> PluginFactoryPtrMap;
  typedef std::pair<yat::IPlugInInfo*, yat::IPlugInFactory*> PluginInfoPair;
  typedef std::pair<IDatasetPtr, IFactoryPtr> DatasetFactoryPair;

private:
  static std::string s_cdma_view;
  static std::string s_dico_path_prop;
  
  PluginMap            m_plugin_map;
  PluginFactoryPtrMap  m_plugin_factory_map;
  cdma::PlugInManager  m_plugin_manager;

  // This is a singleton
  Factory() {}
  
  ~Factory() {}

  // static method to get the unique instance of the class
  static Factory& instance();

  void initPluginMethods(const IFactoryPtr& factory_ptr, Factory::Plugin *plugin_ptr);

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

  /// Get a reference to the factory of the specified plugin
  ///
  /// @param plugin_id   plugin id string (ex.: "SoleilNeXus")
  /// @return            shared pointer on the factory object
  ///
  static IFactoryPtr getPluginFactory(const std::string &plugin_id);
  
  /// Get a pointer or a plugin method
  ///
  /// @param plugin_id   plugin id string (ex.: "SoleilNeXus")
  /// @param method_name the name of a method as it appears in a dictionary mapping document
  /// @return            shared pointer on the factory object 
  ///                    (may be null if the method was not found)
  ///
  static IPluginMethodPtr getPluginMethod(const std::string &plugin_id, 
                                          const std::string &method_name);

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

  /// According to the given factory this method will return the path to reach
  /// the folder containing mapping dictionaries. This file associate entry
  /// keys to paths that are plug-in dependent.
  ///
  /// @param factory_ptr Plug-in instance from which we want to load the dictionary
  /// @return the path to the plug-in's mapping dictionaries folder
  ///
  static std::string getMappingDictionaryFolder(const IFactoryPtr& factory_ptr);

  /// Get the folder path where to search for key dictionary files.
  /// This folder should contains all dictionaries that the above application needs.
  /// @return string path targeting a folder
  ///
  static std::string getDictionariesFolder();

  /// Retrieve the dataset referenced by the string.
  ///
  /// @param uri  yat::URI object of the destination
  /// @return     pair having first: IDataset and second: IFactory
  /// @throw      Exception
  ///
  static DatasetFactoryPair openDataset(const yat::URI& uri) throw ( Exception );

  /// Open a dictionary document
  ///
  /// @param filepath Document file path
  /// @return     Shared pointer on a Dictionary object
  /// @throw      Exception
  ///
  static DictionaryPtr openDictionary(const std::string& filepath) throw ( Exception );

  /// Create an Array with a given data type and shape
  ///
  /// @param shape    shape of the new array
  /// @return         Shared pointer on new Array
  ///
  /// @note Not yet implemented
  ///
  template<typename T> static ArrayPtr createArray(const std::vector<int> shape);

  /// According to the given location (file or folder or what ever)
  /// the factory will try to detect which plugin matches to that destination
  ///
  /// @param location Location of the dataset to analyse
  /// @return IFactoryPtr implementation of a IFactory that fits the data source
  ///
  static IFactoryPtr detectPluginFactory(const yat::URI& location);
};

template<typename T> ArrayPtr Factory::createArray(const std::vector<int> shape)
{
  THROW_NOT_IMPLEMENTED("Factory::createArray");
}

} //namespace CDMACore
#endif //__CDMA_FACTORY_H__

