//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
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
#include <yat/utils/String.h>
#include <yat/system/SysUtils.h>
#include <yat/plugin/PlugInManager.h>
#include <yat/plugin/IPlugInInfo.h>
#include <yat/utils/URI.h>

// CDMA
#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>

namespace cdma
{

typedef IDataItemPtr (*ExternalFunction_t) ( void );

//==============================================================================
/// class IFactory
/// Entry point for cdma client
//==============================================================================
class CDMA_DECL Factory 
{
private:
  struct Plugin
  {
    yat::IPlugInInfo*    info;
    yat::IPlugInFactory* factory;
  };
  
public:
  typedef std::map<std::string, Plugin> PluginMap;
  typedef std::map<std::string, IFactoryPtr> PluginFactoryPtrMap;
  typedef std::pair<yat::IPlugInInfo*, yat::IPlugInFactory*> PluginInfoPair;

private:
  static std::string s_cdma_view;
  static std::string s_dico_path_prop;
  
  PluginMap            m_plugin_map;
  PluginFactoryPtrMap  m_plugin_factory_map;
  yat::PlugInManager   m_plugin_manager;

  // This is a singleton
  Factory() {}
  
  // static method to get the unique instance of the class
  static Factory& instance();

public:

  /// Initialize the factory
  ///
  /// @param 
  static void init(const std::string &plugin_path);
  
  /// Get a reference to the factory of the specified plugin
  ///
  /// @param plugin_id   plugin id string (ex.: "SoleilNeXus")
  /// @return            shared pointer on the factory object
  ///
  static IFactoryPtr getPluginFactory(const std::string &plugin_id);
  
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
  /// @param factory of the plug-in instance from which we want to load the dictionary
  /// @return the path to the plug-in's mapping dictionaries folder
  ///
  static std::string getMappingDictionaryFolder(const IFactoryPtr& factoryName);

  /// Get the folder path where to search for key dictionary files.
  /// This folder should contains all dictionaries that the above application needs.
  /// @return string path targeting a folder
  ///
  static std::string getDictionariesFolder();

  /// Return the logical root group for dictionary mechanism
  /// @return the logical root group
  ///
//  static LogicalGroupPtr getLogicalRootGroup(IDatasetPtr &dataset);
  
  /// Retrieve the dataset referenced by the string.
  ///
  /// @param uri  string object
  /// @return     cdma Dataset
  /// @throw      Exception
  ///
  static IDatasetPtr openDataset(const std::string& uri) throw ( Exception );

  static DictionaryPtr openDictionary(const std::string& filepath) throw ( Exception );

  /// Create an empty Array with a certain data type and certain shape.
  ///
  /// @param clazz   class type
  /// @param shape   array of integer
  /// @return        cdma Array
  ///
  static ArrayPtr createArray(const std::type_info clazz, const std::vector<int> shape);

  /// Create an Array with a given data type, shape and data storage.
  ///
  /// @param clazz    in Class type
  /// @param shape    array of integer
  /// @param storage  a 1D array in the type reference by clazz
  /// @return         cdma Array
  ///
  template<typename T> static ArrayPtr createArray(T type, const std::vector<int> shape);

  /// Create an Array with a given data type, shape and data storage.
  ///
  /// @param clazz    in Class type
  /// @param shape    array of integer
  /// @param storage  a 1D array in the type reference by clazz
  /// @return         cdma Array
  ///
  template<typename T> static ArrayPtr createArray(T* storage, const std::vector<int> shape);

  /// Create an Array from a array. A new 1D array storage will be
  /// created. The new cdma Array will be in the same type and same shape as the
  /// array. The storage of the new array will be a COPY of the supplied
  /// array.
  ///
  /// @param array  one to many dimensional array
  /// @return       cdma Array
  ///
  static ArrayPtr createArray(const void * array);

  /// Create an Array of string storage. The rank of the new Array will be 2
  /// because it treat the Array as 2D char array.
  ///
  /// @param std::string   string value
  /// @return new Array object
  ///
  static ArrayPtr createStringArray(const std::string& value);

  /// Create a double type Array with a given single dimensional double
  /// storage. The rank of the generated Array object will be 1.
  ///
  /// @param  array  double array in one dimension
  /// @return        new Array object
  ///
  static ArrayPtr createDoubleArray(double array[]);

  /// Create a double type Array with a given double storage and shape.
  ///
  /// @param array   double array in one dimension
  /// @param shape   integer array
  /// @return        new Array object
  ///
  static ArrayPtr createDoubleArray(double array[], const std::vector<int> shape);

/*
  /// Create a DataItem with a given cdma parent Group, name and cdma Array data.
  /// If the parent Group is null, it will generate a temporary Group as the
  /// parent group.
  ///
  /// @param parent     cdma Group
  /// @param shortName  in string type
  /// @param array      cdma Array
  /// @return           cdma IDataItem
  /// @throw            Exception
  ///
  static IDataItemPtr createDataItem(const IGroupPtr& parent, const std::string& shortName, const ArrayPtr& array) throw ( Exception );

  /// Create a cdma Group with a given parent cdma Group, name, and a bool
  /// initiate parameter telling the factory if the new group will be put in
  /// the list of children of the parent. Group.
  ///
  /// @param parent        cdma Group
  /// @param shortName     in string type
  /// @param updateParent  if the parent will be updated
  /// @return              cdma Group
  ///
  static IGroupPtr createGroup(const IGroupPtr& parent, const std::string& shortName, const bool updateParent);

  /// Create an empty cdma Group with a given name. The factory will create an
  /// empty cdma Dataset first, and create the new Group under the root group of
  /// the Dataset.
  ///
  /// @param shortName     in string type
  /// @return              cdma Group
  /// @throw               Exception
  ///
  static IGroupPtr createGroup(const std::string& shortName) throw ( Exception );

  /// Create a cdma Attribute with given name and value.
  ///
  /// @param name          in string type
  /// @param value         in string type
  /// @return              cdma Attribute
  ///
  static IAttributePtr createAttribute(const std::string& name, const void * value);

  /// Create a cdma Dataset with a string reference. If the file exists, it will
  ///
  /// @param uri           string object
  /// @return              cdma Dataset
  /// @throw               Exception
  ///
  static IDatasetPtr createDatasetInstance(const std::string& uri) throw ( Exception );

  /// Create a cdma Dataset in memory only. The dataset is not open yet. It is
  /// necessary to call dataset.open() to access the root of the dataset.
  ///
  /// @return a cdma Dataset
  /// @throw  Exception
  ///
  static IDatasetPtr createEmptyDatasetInstance() throw ( Exception );
*/
  static KeyPtr createKey(std::string keyName);

  static PathPtr createPath( std::string path );

  static PathParameterPtr createPathParameter(CDMAType::ParameterType type, std::string& name, void * value);

  static IPathParamResolverPtr createPathParamResolver(const PathPtr& path);

  /// According to the given destination (file or folder or what ever)
  /// the factory will try to detect which plugin matches to that destination
  ///
  /// @return IFactoryPtr implementation of a IFactory that fits the data source
  ///
  static IFactoryPtr detectPluginFactory(const yat::URI& destination);

};

template<typename T> ArrayPtr Factory::createArray(T type, const std::vector<int> shape) {}
template<typename T> ArrayPtr Factory::createArray(T* storage, const std::vector<int> shape) {}

} //namespace CDMACore
#endif //__CDMA_FACTORY_H__

