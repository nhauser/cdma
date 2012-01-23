//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
// Contributors :
// See AUTHORS file 
//******************************************************************************
#ifndef __IOBJECT_H__
#define __IOBJECT_H__

#include <cdma/Common.h>
#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>
#include <list>

namespace cdma
{

class CDMA_DECL CDMAType
{
  public:
  //==============================================================================
  /// ModelType
  /// Kind of CDMA objects
  //==============================================================================
  enum ModelType
  {
    Group = 0,
    DataItem = 1,
    Attribute = 2,
    Dimension = 3,
    Array = 4,
    Dataset = 5,
    LogicalGroup = 6,
    Dictionary = 7,
    Other = 8
  };

  //==============================================================================
  /// ParameterType
  /// <To be completed>
  //==============================================================================
  enum ParameterType
  {
    Substitution = 1
  };
};

// Forward declaration
class Array;
class ArrayIterator;
class View;
class Range;
class SliceIterator;
class Slicer;
class ArrayUtils;
class ArrayMath;
class IArrayStorage;
class IAttribute;
class IClassLoader;
class IContainer;
class IContext;
class IDataItem;
class IDataset;
class IDataSource;
class IDimension;
class IFactory;
class IFactoryManager;
class IFactoryResolver;
class IGroup;
class IPathMethod;
class IPathParamResolver;
class LogicalGroup;
class Key;
class Path;
class PathParameter;
class Dictionary;

/// Shared pointers declaration
typedef yat::SharedPtr<Array, yat::Mutex> ArrayPtr;
typedef yat::SharedPtr<View, yat::Mutex> ViewPtr;
typedef yat::SharedPtr<Range, yat::Mutex> RangePtr;
typedef yat::SharedPtr<ArrayIterator, yat::Mutex> ArrayIteratorPtr;
typedef yat::SharedPtr<SliceIterator, yat::Mutex> SliceIteratorPtr;
typedef yat::SharedPtr<Slicer, yat::Mutex> SlicerPtr;
typedef yat::SharedPtr<ArrayUtils, yat::Mutex> ArrayUtilsPtr;
typedef yat::SharedPtr<ArrayMath, yat::Mutex> ArrayMathPtr;
typedef yat::SharedPtr<IArrayStorage, yat::Mutex> IArrayStoragePtr;
typedef yat::SharedPtr<IAttribute, yat::Mutex> IAttributePtr;
typedef yat::SharedPtr<IClassLoader, yat::Mutex> IClassLoaderPtr;
typedef yat::SharedPtr<IContext, yat::Mutex> IContextPtr;
typedef yat::SharedPtr<IDataItem, yat::Mutex> IDataItemPtr;
typedef yat::SharedPtr<IDataset, yat::Mutex> IDatasetPtr;
typedef yat::SharedPtr<IDataSource, yat::Mutex> IDataSourcePtr;
typedef yat::SharedPtr<IDimension, yat::Mutex> IDimensionPtr;
typedef yat::SharedPtr<IFactory, yat::Mutex> IFactoryPtr;
typedef yat::SharedPtr<IFactoryManager, yat::Mutex> IFactoryManagerPtr;
typedef yat::SharedPtr<IFactoryResolver, yat::Mutex> IFactoryResolverPtr;
typedef yat::SharedPtr<IGroup, yat::Mutex> IGroupPtr;
typedef yat::SharedPtr<IPathMethod, yat::Mutex> IPathMethodPtr;
typedef yat::SharedPtr<IPathParamResolver, yat::Mutex> IPathParamResolverPtr;
typedef yat::SharedPtr<LogicalGroup, yat::Mutex> LogicalGroupPtr;
typedef yat::SharedPtr<Key, yat::Mutex> KeyPtr;
typedef yat::SharedPtr<Path, yat::Mutex> PathPtr;
typedef yat::SharedPtr<PathParameter, yat::Mutex> PathParameterPtr;
typedef yat::SharedPtr<Dictionary, yat::Mutex> DictionaryPtr;

/// Generic types
typedef std::list<std::string> StringList;
typedef yat::SharedPtr<StringList, yat::Mutex> StringListPtr;

//==============================================================================
/// IObject
/// Base interface of all CDMA classes
//==============================================================================
class CDMA_DECL IObject
{
 public:
  virtual ~IObject() {}
  
  /// Get the name of the factory that can create this item
  virtual std::string getFactoryName() const = 0;

  /// Get the ModelType implemented by this object.
  virtual CDMAType::ModelType getModelType() const = 0; 
};

} // namespace

#endif // __IOBJECT_H__
