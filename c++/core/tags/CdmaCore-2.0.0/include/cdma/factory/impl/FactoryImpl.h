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

#ifndef __CDMA_FACTORY_IMPL_H__
#define __CDMA_FACTORY_IMPL_H__

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
#include <cdma/factory/Factory.h>
#include <cdma/exception/Exception.h>
#include <cdma/factory/plugin/IPluginFactory.h>
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/dictionary/plugin/PluginMethods.h>
#include <cdma/array/IArray.h>
#include <cdma/array/impl/Array.h>

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
class PlugInManager: public yat::PlugInManager { };

//==============================================================================
/// @brief Entry point for CDMA client
///
/// Client application get access to datasets from static methods of this class
//==============================================================================
class FactoryImpl
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
  typedef std::map<std::string, IPluginFactoryPtr> PluginFactoryPtrMap;
  typedef std::pair<yat::IPlugInInfo*, yat::IPlugInFactory*> PluginInfoPair;

private:
  static std::string s_cdma_view;
  static std::string s_dico_path_prop;
  
  PluginMap                m_plugin_map;
  PluginFactoryPtrMap      m_plugin_factory_map;
  PlugInManager            m_plugin_manager;
  Dictionary::IdConceptMap m_core_dict;

  // This is a singleton
  FactoryImpl() {}
  
  ~FactoryImpl() {}

  // static method to get the unique instance of the class
  static FactoryImpl& instance();

  void initPluginMethods(const IPluginFactoryPtr& factory_ptr, FactoryImpl::Plugin *plugin_ptr);

public:

  /// Initialize the factory
  static void init(const std::string &plugin_path);
  
  /// Clean up memory
  static void cleanup();

  /// Get a reference to the factory of the specified plugin
  static IPluginFactoryPtr getPluginFactory(const std::string &plugin_id);
  
  /// Get a pointer or a plugin method
  static IPluginMethodPtr getPluginMethod(const std::string &plugin_id, 
                                          const std::string &method_name);

  static void setActiveView( const std::string& experiment);

  static const std::string& getActiveView();

  /// According to the currently defined experiment, this method will return the path
  static std::string getKeyDictionaryPath();

  /// This method returns the path of the data definition documents.
  static std::string getKeyDictionaryFolder();

  /// According to the given factory this method will return the path to reach
  static std::string getMappingDictionaryFolder(const std::string& plugin_id);

  /// This method returns the path of the concepts dictionaries.
  static std::string getConceptDictionaryFolder();

  /// Get the folder path where to search for key dictionary files.
  static std::string getDictionariesFolder();

  /// Retrieve the dataset referenced by the string.
  static IDatasetPtr openDataset(const yat::URI& uri) throw ( Exception );

  /// Open a dictionary document
  static DictionaryPtr openDictionary(const std::string& filepath) throw ( Exception );

  /// Create an Array with a given data type and shape
  template<typename T> static IArrayPtr createArray(const std::vector<int> shape);

  /// According to the given location (file or folder or what ever)
  /// the factory will try to detect which plugin matches to that destination
  static IPluginFactoryPtr detectPluginFactory(const yat::URI& location) throw ( Exception );

  /// According to a dataset location the factory will try to returns the datasource object
  /// that belong to the most suitable plugin
  static IDataSourcePtr getDataSource(const yat::URI& location) throw ( Exception );
};

/// @endcond

} //namespace CDMACore
#endif //__CDMA_FACTORY_H__

