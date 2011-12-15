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
// #include <yat/utils/URI.h>
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
  
  /// Create an empty Array with a certain data type and certain shape.
  ///
  /// @param clazz Class type
  /// @param shape array of integer
  /// @return CDMA Array Created on 18/06/2008
  ///
  virtual IArrayPtr createArray(const std::type_info clazz, const std::vector<int> shape) = 0;

  /// Create an Array with a given data type, shape and data storage.
  ///
  /// @param clazz   in Class type
  /// @param shape   array of integer
  /// @param storage a 1D  array in the type reference by clazz
  /// @return CDMA Array Created on 18/06/2008
  ///
  virtual IArrayPtr createArray(const std::type_info clazz, const std::vector<int> shape, const void * storage) = 0;

  /// Create an Array from a  array. A new 1D  array storage will be
  /// created. The new CDMA Array will be in the same type and same shape as the
  ///  array. The storage of the new array will be a COPY of the supplied
  ///  array.
  ///
  /// @param array one to many dimensional  array
  /// @return CDMA Array Created on 18/06/2008
  ///
  virtual IArrayPtr createArray(const void * array) = 0;

  /// Create an Array of string storage. The rank of the new Array will be 2
  /// because it treat the Array as 2D char array.
  ///
  /// @param string string value
  /// @return new Array object
  ///
  virtual IArrayPtr createStringArray(const std::string& value) = 0;

  /// Create a double type Array with a given single dimensional  double
  /// storage. The rank of the generated Array object will be 1.
  ///
  /// @param array double array in one dimension
  /// @return new Array object Created on 10/11/2008
  ///
  virtual IArrayPtr createDoubleArray(double array[]) = 0;

  /// Create a double type Array with a given  double storage and shape.
  ///
  /// @param array double array in one dimension
  /// @param shape integer array
  /// @return new Array object Created on 10/11/2008
  ///
  virtual IArrayPtr createDoubleArray(double array[], const std::vector<int> shape) = 0;

  /// Create an IArray from a  array. A new 1D  array storage will be
  /// created. The new CDMA Array will be in the same type and same shape as the
  ///  array. The storage of the new array will be the supplied  array.
  ///
  /// @param array primary array
  /// @return CDMA array Created on 28/10/2008
  ///
  virtual IArrayPtr createArrayNoCopy(const void * array) = 0;

  /// Create a DataItem with a given CDMA parent Group, name and CDMA Array data.
  /// If the parent Group is null, it will generate a temporary Group as the
  /// parent group.
  ///
  /// @param parent    CDMA Group
  /// @param shortName in string type
  /// @param array     CDMA Array
  /// @return CDMA     IDataItem
  /// @throw  Exception
  ///
  virtual IDataItemPtr createDataItem(const IGroupPtr& parent, const std::string& shortName, const IArrayPtr& array) throw ( Exception ) = 0;

  /// Create a CDMA Group with a given parent CDMA Group, name, and a bool
  /// initiate parameter telling the factory if the new group will be put in
  /// the list of children of the parent. Group.
  ///
  /// @param parent       CDMA Group
  /// @param shortName    in string type
  /// @param updateParent if the parent will be updated
  /// @return CDMA Group Created on 18/06/2008
  ///
  virtual IGroupPtr createGroup(const IGroupPtr& parent, const std::string& shortName, const bool updateParent) = 0;

  /// Create an empty CDMA Group with a given name. The factory will create an
  /// empty CDMA Dataset first, and create the new Group under the root group of
  /// the Dataset.
  ///
  /// @param shortName
  ///            in string type
  /// @return CDMA Group
  /// @throw  Exception
  ///
  virtual IGroupPtr createGroup(const std::string& shortName) throw ( Exception ) = 0;

  /// Create a CDMA Attribute with given name and value.
  ///
  /// @param name  in string type
  /// @param value in string type
  /// @return CDMA Attribute Created on 18/06/2008
  ///
  virtual IAttributePtr createAttribute(const std::string& name, const void * value) = 0;

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

  virtual KeyPtr createKey( std::string keyName ) = 0;

  virtual PathPtr createPath( std::string path ) = 0;

  virtual PathParameterPtr createPathParameter( CDMAType::ParameterType type, const std::string& name, void * value ) = 0;

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
};

} //namespace CDMACore
#endif //__CDMA_IFACTORY_H__

