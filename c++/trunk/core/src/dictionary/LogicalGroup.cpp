//*****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez Clément
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
//*****************************************************************************

//-----------------------------------------------------------------------------
// DEPENDENCIES
//-----------------------------------------------------------------------------

#include <yat/utils/String.h>

#include <cdma/exception/Exception.h>
#include <cdma/dictionary/LogicalGroup.h>
#include <cdma/dictionary/Dictionary.h>
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/Path.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/navigation/IDataItem.h>

namespace cdma
{
  //-----------------------------------------------------------------------------
  // LogicalGroup::LogicalGroup
  //-----------------------------------------------------------------------------
  LogicalGroup::LogicalGroup( IDataset* pDataset, LogicalGroup* pParent, const KeyPtr& pKey, const DictionaryPtr& pDictionary )
  {
    m_pDataset = pDataset;
    m_pParent = pParent;
    m_keyPtr = pKey;
    m_dictionaryPtr = pDictionary;
  }


  //-----------------------------------------------------------------------------
  // LogicalGroup::~LogicalGroup
  //-----------------------------------------------------------------------------
  LogicalGroup::~LogicalGroup()
  {
    
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getDataItem
  //-----------------------------------------------------------------------------
  IDataItemPtr LogicalGroup::getDataItem(const KeyPtr& key)
  {
    IDataItemPtr item;
    if( ! key.is_null() )
    {
      PathPtr path = m_dictionaryPtr->getPath(key);
      yat::String pathValue;
      if( ! path.is_null() )
      {
        pathValue = path->getValue();
      }
      item = m_pDataset->getItemFromPath(pathValue);
      //item->setLocation(pathValue);
      item->setShortName( key->getName() );
    }
    return item;
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getDataItem
  //-----------------------------------------------------------------------------
  IDataItemPtr LogicalGroup::getDataItem(const std::string& keyPath)
  {
    THROW_NOT_IMPLEMENTED("LogicalGroup::getDataItem");
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getDataItemList
  //-----------------------------------------------------------------------------
  std::list<IDataItemPtr> LogicalGroup::getDataItemList(const KeyPtr& key)
  {
    THROW_NOT_IMPLEMENTED("LogicalGroup::getDataItemList");
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getDataItemList
  //-----------------------------------------------------------------------------
  std::list<IDataItemPtr> LogicalGroup::getDataItemList(const std::string& keyPath)
  {
    THROW_NOT_IMPLEMENTED("LogicalGroup::getDataItemList");
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getGroup
  //-----------------------------------------------------------------------------
  LogicalGroupPtr LogicalGroup::getGroup(const KeyPtr& key)
  {
    CDMA_DBG("[BEGIN] LogicalGroup::getGroup");
    // Check key isn't empty
    LogicalGroupPtr child;
    if( ! key.is_null() )
    {
      // Store key's name
      yat::String keyName = key->getName();

      // Check key hasn't beed asked yet
      std::map<std::string, LogicalGroupPtr>::iterator itChild = m_childGroups.find( keyName );
      if( itChild != m_childGroups.end() )
      {
        child = itChild->second;
      }
      // Check if key is in children list
      else
      {
        yat::String groupKeyName = m_keyPtr.is_null() ? "" : m_keyPtr->getName();
        if( m_listKeyPtr.is_null() )
        {
          m_listKeyPtr = m_dictionaryPtr->getKeys( groupKeyName );
        }
        StringList::iterator itChildren;
        for( itChildren = m_listKeyPtr->begin(); itChildren != m_listKeyPtr->end(); itChildren++ )
        {
          // Key is in children list -> construct group
          if( *itChildren == keyName )
          {
            child = new LogicalGroup(m_pDataset, this, key, m_dictionaryPtr );
            m_childGroups[ keyName ] = child;
            break;
          }
        }
      }
    }
    CDMA_DBG("[END] LogicalGroup::getGroup");
    return child;
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getGroup
  //-----------------------------------------------------------------------------
  LogicalGroupPtr LogicalGroup::getGroup(const std::string& keyPath)
  {
    THROW_NOT_IMPLEMENTED("LogicalGroup::getGroup");
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getKeyNames
  //-----------------------------------------------------------------------------
  std::list<std::string> LogicalGroup::getKeyNames(CDMAType::ModelType model)
  {
    THROW_NOT_IMPLEMENTED("LogicalGroup::getKeyNames");
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getKeys
  //-----------------------------------------------------------------------------
  std::list<KeyPtr> LogicalGroup::getKeys()
  {
    CDMA_DBG("[BEGIN] LogicalGroup::getKeys");
    std::list< KeyPtr > result;

    // Get children keys' names
    yat::String key = m_keyPtr.is_null() ? "" : m_keyPtr->getName();
    if( m_listKeyPtr.is_null() )
    {
      m_listKeyPtr = m_dictionaryPtr->getKeys( key );
    }
    for( StringList::iterator itChildren = m_listKeyPtr->begin(); itChildren != m_listKeyPtr->end(); itChildren++ )
    {
      result.push_back( KeyPtr(new Key( *itChildren, m_dictionaryPtr->getKeyType(*itChildren)) ) );
    }

    CDMA_DBG("[BEGIN] LogicalGroup::getKeys");
    return result;
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::bindKey
  //-----------------------------------------------------------------------------
  KeyPtr LogicalGroup::bindKey(const std::string& bind, const KeyPtr& key)
  {
    THROW_NOT_IMPLEMENTED("LogicalGroup::bindKey");
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getParameterValues
  //-----------------------------------------------------------------------------
  std::list<PathParameterPtr> LogicalGroup::getParameterValues(const KeyPtr& key)
  {
    THROW_NOT_IMPLEMENTED("LogicalGroup::getParameterValues");
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::setParent
  //-----------------------------------------------------------------------------
  void LogicalGroup::setParent(LogicalGroup& group)
  {
    m_pParent = &group;
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::setParent
  //-----------------------------------------------------------------------------
  LogicalGroupPtr LogicalGroup::getParent()
  {
    return m_pParent;
  }
  
  //-----------------------------------------------------------------------------
  // LogicalGroup::setParent
  //-----------------------------------------------------------------------------
  std::string LogicalGroup::getLocation()
  {
    yat::String path = getName();
    LogicalGroupPtr parent = getParent();
    while( ! parent.is_null() )
    {
      path = parent->getName() + (parent->getName() == "" ? "" : "/" ) + path;
      parent = parent->getParent();
    }
    return path;
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::setParent
  //-----------------------------------------------------------------------------
  std::string LogicalGroup::getName()
  {
    return m_keyPtr.is_null() ? "" : m_keyPtr->getName();
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::setParent
  //-----------------------------------------------------------------------------
  std::string LogicalGroup::getShortName()
  {
    return getName();
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getFactoryName
  //-----------------------------------------------------------------------------
  std::string LogicalGroup::getFactoryName() const
  {
    return m_pDataset->getFactoryName();
  }

  //-----------------------------------------------------------------------------
  // LogicalGroup::getModelType
  //-----------------------------------------------------------------------------
  cdma::CDMAType::ModelType LogicalGroup::getModelType() const
  {
    return cdma::CDMAType::LogicalGroup;
  }
}
