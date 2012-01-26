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

#define DECLARE_SHARED_PTR(x)\
  typedef yat::SharedPtr<x, yat::Mutex> x##Ptr

#define DECLARE_CLASS_SHARED_PTR(x)\
  class x;\
  DECLARE_SHARED_PTR(x)

#define DECLARE_WEAK_PTR(x)\
  typedef yat::WeakPtr<x, yat::Mutex> x##WPtr

#define DECLARE_CLASS_WEAK_PTR(x)\
  class x;\
  DECLARE_WEAK_PTR(x)

#define DECLARE_SHARED_WEAK_PTR(x)\
  DECLARE_SHARED_PTR(x);\
  DECLARE_WEAK_PTR(x)

#define DECLARE_CLASS_SHARED_WEAK_PTR(x)\
  class x;\
  DECLARE_SHARED_PTR(x);\
  DECLARE_WEAK_PTR(x)

// Forward declarations
DECLARE_CLASS_SHARED_WEAK_PTR(Array);
DECLARE_CLASS_SHARED_PTR(ArrayIterator);
DECLARE_CLASS_SHARED_PTR(ArrayUtils);
DECLARE_CLASS_SHARED_PTR(ArrayMath);
DECLARE_CLASS_SHARED_PTR(Dictionary);
DECLARE_CLASS_SHARED_PTR(IArrayStorage);
DECLARE_CLASS_SHARED_PTR(IAttribute);
DECLARE_CLASS_SHARED_PTR(IClassLoader);
DECLARE_CLASS_SHARED_PTR(IContainer);
DECLARE_CLASS_SHARED_PTR(IContext);
DECLARE_CLASS_SHARED_PTR(IDataItem);
DECLARE_CLASS_SHARED_PTR(IDataset);
DECLARE_CLASS_SHARED_PTR(IDataSource);
DECLARE_CLASS_SHARED_PTR(IDimension);
DECLARE_CLASS_SHARED_PTR(IFactory);
DECLARE_CLASS_SHARED_PTR(IFactoryManager);
DECLARE_CLASS_SHARED_PTR(IFactoryResolver);
DECLARE_CLASS_SHARED_PTR(IGroup);
DECLARE_CLASS_SHARED_PTR(IPathMethod);
DECLARE_CLASS_SHARED_PTR(IPathParamResolver);
DECLARE_CLASS_SHARED_PTR(Key);
DECLARE_CLASS_SHARED_PTR(LogicalGroup);
DECLARE_CLASS_SHARED_PTR(Path);
DECLARE_CLASS_SHARED_PTR(PathParameter);
DECLARE_CLASS_SHARED_PTR(SliceIterator);
DECLARE_CLASS_SHARED_PTR(Slicer);
DECLARE_CLASS_SHARED_PTR(Range);
DECLARE_CLASS_SHARED_PTR(View);

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
