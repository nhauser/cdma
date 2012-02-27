//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_CONTEXT_H__
#define __CDMA_CONTEXT_H__

#include <vector>

// YAT library
#include <yat/memory/SharedPtr.h>

// Include CDMA
#include <cdma/Common.h>
#include <cdma/navigation/IContainer.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/Dictionary.h>

namespace cdma
{

//==============================================================================
/// This interface is used when invoking an external method.
/// It should contain all required information so the called method,
/// can work properly as if it were in the CDM.
/// The context is compound of the dataset we are working on,
/// the caller of the method, the key used to call that method (that can
/// have some parameters), the path (with parameters set) and some
/// parameters that are set by the institute's plug-in.
//==============================================================================
class Context
{
private:
  KeyPtr          m_key_ptr;
  IDataset*       m_dataset_ptr;
  DataItemList    m_dataitems;
  DictionaryPtr   m_dictionary_ptr;

public:
  
  /// Constructor
  Context() {}

  /// Constructor
  Context(IDataset* dataset_ptr, const KeyPtr& key_ptr, const DictionaryPtr dict_ptr)
    : m_key_ptr(key_ptr), m_dataset_ptr(dataset_ptr), m_dictionary_ptr(dict_ptr)  {}

  //@{ Accessors

  /// Return smart pointer on key object
  const KeyPtr& getKey() const { return m_key_ptr; }

  /// Return smart pointer on key object
  const DictionaryPtr& getDictionary() const { return m_dictionary_ptr; }

  /// Return smart pointer on dataset object
  // @warning non-const method!
  IDataset* getDataset() { return m_dataset_ptr; }

  /// Get read/write access on the data items list
  const DataItemList& getDataItems() const { return m_dataitems; }

  /// Get read/write access on the data items list
  IDataItemPtr getTopDataItem() const;

  //@}

  /// Push a data items at the back of the list
  void pushDataItem(const IDataItemPtr& dataitem_ptr);

  /// Clear the data items list
  void clearDataItems()
  {
    m_dataitems.clear();
  }
};

} //namespace CDMACore
#endif //__CDMA_CONTEXT_H__

