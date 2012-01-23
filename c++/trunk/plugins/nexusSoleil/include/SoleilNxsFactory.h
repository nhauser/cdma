#ifndef __CDMA_NXSFACTORY_H__
#define __CDMA_NXSFACTORY_H__

#include <vector>
#include <string>

// YAT library
#include <yat/plugin/PlugInSymbols.h>
// #include <yat/utils/URI.h>
#include <yat/plugin/IPlugInInfo.h>

// CDMA core
#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>
#include <cdma/IFactory.h>
#include <cdma/dictionary/Key.h>

// NeXus engine
#include <NxsDataset.h>

// Soleil NeXus plug-in
#include <SoleilNxsFactory.h>

namespace cdma
{

//==============================================================================
/// Plugin info class
//==============================================================================
class SoleilNxsFactoryInfo : public yat::IPlugInInfo
{
public:
  virtual std::string get_plugin_id(void) const;
  virtual std::string get_interface_name(void) const;
  virtual std::string get_version_number(void) const;
};

//==============================================================================
/// IFactory implementation
//==============================================================================
class SoleilNxsFactory : public IFactory 
{
public:
  SoleilNxsFactory() {};
  ~SoleilNxsFactory() {};

  /// Retrieve the dataset referenced by the string.
  ///
  /// @param uri string object
  /// @return IDataset
  ///
  IDatasetPtr openDataset(const std::string& path) throw ( cdma::Exception );
  // IDatasetPtr openDataset(const yat::URI& uri) throw ( cdma::Exception );

  DictionaryPtr openDictionary(const std::string& filepath) throw ( cdma::Exception );

  /// Create an empty Array with a certain data type and certain shape.
  ///
  /// @param clazz Class type
  /// @param shape vector of integer describing the shape
  /// @return Array
  ///
  ArrayPtr createArray(const std::type_info clazz, const std::vector<int> shape);

  /// Create an Array with a given data type, shape and data storage.
  ///
  /// @param clazz in Class type
  /// @param shape array of integer
  /// @param storage a 1D  array in the type reference by clazz
  /// @return Array
  ///
  ArrayPtr createArray(const std::type_info clazz, const std::vector<int> shape, const void * storage);

  /// Create an Array from an array. A new 1D array storage will be
  /// created. The Array will be in the same type and same shape as the
  /// given array. The storage of the new array will be a COPY of the supplied
  /// array.
  ///
  /// @param array one to many dimensional java array
  /// @return Array
  ///
  ArrayPtr createArray(const void * array);

  /// Create an Array of string storage. The rank of the new Array will be 2
  /// because it treat the Array as 2D char array.
  ///
  /// @param string string value
  /// @return new Array object
  ///
  ArrayPtr createStringArray(const std::string& value);

  /// Create a double type Array with a given single dimensional double
  /// storage. The rank of the generated Array object will be 1.
  ///
  /// @param array array of double in one dimension
  /// @return new Array object
  ///
  ArrayPtr createDoubleArray(double array[]);

  /// Create a double type Array with a given java double storage and shape.
  ///
  /// @param array array of double in one dimension
  /// @param shape integer vector
  /// @return new Array object
  ///
  ArrayPtr createDoubleArray(double array[], const std::vector<int> shape);

  /// Create an Array from a array. The new Array will be in the same type and same shape as the
  /// given array. The storage of the new array will be the supplied array.
  ///
  /// @param array primary array
  /// @return Array
  ///
  ArrayPtr createArrayNoCopy(const void * array);

  /// Create a DataItem with a given parent Group, name and Array data.
  /// If the parent Group is null, it will generate a temporary Group as the
  /// parent group.
  ///
  /// @param parent IGroup
  /// @param shortName in string type
  /// @param array Array
  /// @return an IDataItem
  ///
  IDataItemPtr createDataItem(const cdma::IGroupPtr& parent, const std::string& shortName, const cdma::ArrayPtr& array) throw ( cdma::Exception );

  /// Create a GDM Group with a given parent GDM Group, name, and a bool
  /// initiate parameter telling the factory if the new group will be put in
  /// the list of children of the parent. Group.
  ///
  /// @param parent IGroup
  /// @param shortName in string type
  /// @param updateParent if the parent will be updated
  /// @return IGroup
  ///
  IGroupPtr createGroup(const cdma::IGroupPtr& parent, const std::string& shortName, const bool updateParent);

  /// Create an empty GDM Group with a given name. The factory will create an
  /// empty IDataset first, and create the new IGroup under the root group of
  /// the Dataset.
  ///
  /// @param shortName in string type
  /// @return IGroup
  ///
  IGroupPtr createGroup(const std::string& shortName) throw ( cdma::Exception );

  /// Create a IAttribute with given name and value.
  ///
  /// @param name in string type
  /// @param value in string type
  /// @return IAttribute
  ///
  IAttributePtr createAttribute(const std::string& name, const void * value);

  /// Create a IDataset with a string reference for destination file
  ///
  /// @param uri string object
  /// @return IDataset
  ///
  IDatasetPtr createDatasetInstance(const std::string& uri) throw ( cdma::Exception );

  /// Create a GDM Dataset in memory only. The dataset is not open yet. It is
  /// necessary to call dataset.open() to access the root of the dataset.
  ///
  /// @return a IDataset
  ///
  IDatasetPtr createEmptyDatasetInstance() throw ( cdma::Exception );

  KeyPtr createKey( std::string keyName );

  PathPtr createPath( std::string path );

  PathParameterPtr createPathParameter(cdma::CDMAType::ParameterType type, const std::string& name, void * value);

  IPathParamResolverPtr createPathParamResolver(const cdma::PathPtr& path);


  /// Return the symbol used by the plug-in to separate nodes in a string path
  /// @note <b>EXPERIMENTAL METHOD</b> do note use/implements
  ///
  std::string getPathSeparator();


  /// The factory has a unique name that identifies it.
  /// @return the factory's name
  ///
  std::string getName() { return NXS_FACTORY_NAME; };
  
  /// Returns the plugin's IDataSource implementation  
  IDataSourcePtr getPluginURIDetector();
};
} //namespace CDMACore
#endif //__CDMA_IFACTORY_H__

