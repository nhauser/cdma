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

//-----------------------------------------------------------------------------
// DEPENDENCIES
//-----------------------------------------------------------------------------

#include <yat/utils/String.h>

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/LogicalGroup.h>
#include <cdma/dictionary/Dictionary.h>
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/Context.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/navigation/IDataItem.h>

namespace cdma
{

//-----------------------------------------------------------------------------
// LogicalGroup::LogicalGroup
//-----------------------------------------------------------------------------
LogicalGroup::LogicalGroup( IDataset* dataset_ptr, LogicalGroup* parent_ptr, 
                            const KeyPtr& key_ptr, const DictionaryPtr& dictionary_ptr )
{
  m_dataset_ptr = dataset_ptr;
  if( parent_ptr != NULL )
  {
    m_parent_path = parent_ptr->getLocation();
  }
  m_key_ptr = key_ptr;
  m_dictionary_ptr = dictionary_ptr;
}


//-----------------------------------------------------------------------------
// LogicalGroup::~LogicalGroup
//-----------------------------------------------------------------------------
LogicalGroup::~LogicalGroup()
{
  
}

//-----------------------------------------------------------------------------
// LogicalGroup::PrivSolveKey
//-----------------------------------------------------------------------------
void LogicalGroup::PrivSolveKey(Context *context_ptr)
{
  CDMA_FUNCTION_TRACE("LogicalGroup::PrivSolveKey");

  KeyPtr key_ptr = context_ptr->getKey();

  if( key_ptr )
  {
    try
    {
      SolverList solvers = m_dictionary_ptr->getSolversList(key_ptr);

      // Iterate on the solvers list
      for( SolverList::iterator it = solvers.begin(); it != solvers.end(); it++ )
      {
        (*it)->solve(*context_ptr);
      }
    }
    catch( cdma::Exception &ex )
    {
      ex.push_error( "KEY_ERROR", PSZ_FMT( "Unable to get data from key '%s'",
                                           PSZ( key_ptr->getName() ) ),
                                           "LogicalGroup::PrivSolveKey" );
      throw ex;
    }
  }
  else
  {
    throw cdma::Exception( "NULL_POINTER",
                           "A valid key is required",
                           "LogicalGroup::PrivSolveKey" );
  }
}

//-----------------------------------------------------------------------------
// LogicalGroup::getDataItem
//-----------------------------------------------------------------------------
IDataItemPtr LogicalGroup::getDataItem(const KeyPtr& key_ptr)
{
  CDMA_FUNCTION_TRACE("LogicalGroup::getDataItem");
  CDMA_TRACE("key: " << key_ptr->getName());

  try
  {
    Context context(m_dataset_ptr, key_ptr, m_dictionary_ptr);
    PrivSolveKey(&context);
    IDataItemPtr item = context.getTopDataItem();
    return item;
  }
  catch( cdma::Exception& )
  {
    throw;
  }
}

//-----------------------------------------------------------------------------
// LogicalGroup::getDataItem
//-----------------------------------------------------------------------------
IDataItemPtr LogicalGroup::getDataItem(const std::string& keypath)
{
  if( m_dataset_ptr != NULL )
  {
    yat::String path = keypath;
    
    std::vector<yat::String> keys;
    path.replace( "::", "/" );
    path.split( '/', &keys );
    
    LogicalGroupPtr tmp = m_dataset_ptr->getLogicalRoot();
    
    for( unsigned int i = 0; i < keys.size() - 1 && ! tmp.is_null(); i++ )
    {
       tmp = tmp->getGroup( KeyPtr( new Key( keys[i], Key::GROUP ) ) );
    }
    
    if( tmp )
    {
      return tmp->getDataItem( KeyPtr( new Key( keys[ keys.size() - 1 ], Key::ITEM ) ) );
    }
    else
    {
      return NULL;
    }
  }
  return NULL;
}

//-----------------------------------------------------------------------------
// LogicalGroup::getDataItemList
//-----------------------------------------------------------------------------
std::list<IDataItemPtr> LogicalGroup::getDataItemList(const KeyPtr& key_ptr)
{
  try
  {
    IDataItemPtr item;
    Context context(m_dataset_ptr, key_ptr, m_dictionary_ptr);
    PrivSolveKey(&context);
    return context.getDataItems();
  }
  catch( cdma::Exception& )
  {
    throw;
  }
}

//-----------------------------------------------------------------------------
// LogicalGroup::getDataItemList
//-----------------------------------------------------------------------------
DataItemList LogicalGroup::getDataItemList(const std::string& keyword)
{
  return getDataItemList( KeyPtr( new Key( keyword, Key::ITEM ) ) );
}

//-----------------------------------------------------------------------------
// LogicalGroup::getGroup
//-----------------------------------------------------------------------------
LogicalGroupPtr LogicalGroup::getGroup(const KeyPtr& key_ptr)
{
  CDMA_FUNCTION_TRACE("LogicalGroup::getGroup");
  // Check key isn't empty
  LogicalGroupPtr child;
  if( key_ptr )
  {
    // Store key's name
    yat::String keyName = key_ptr->getName();

    // Check key hasn't beed asked yet
    std::map<std::string, LogicalGroupPtr>::iterator itChild = m_child_groups.find( keyName );
    if( itChild != m_child_groups.end() )
    {
      child = itChild->second;
    }
    // Check if key is in children list
    else
    {
      yat::String groupKeyName = m_key_ptr.is_null() ? "" : m_key_ptr->getName();
      if( m_listkey_ptr.is_null() )
      {
        m_listkey_ptr = m_dictionary_ptr->getKeys( groupKeyName );
      }
      StringList::iterator itChildren;
      for( itChildren = m_listkey_ptr->begin(); itChildren != m_listkey_ptr->end(); itChildren++ )
      {
        // Key is in children list -> construct group
        if( *itChildren == keyName )
        {
          child = new LogicalGroup(m_dataset_ptr, this, key_ptr, m_dictionary_ptr );
          m_child_groups[ keyName ] = child;
          break;
        }
      }
    }
  }
  return child;
}

//-----------------------------------------------------------------------------
// LogicalGroup::getGroup
//-----------------------------------------------------------------------------
LogicalGroupPtr LogicalGroup::getGroup(const std::string& keypath)
{
  // TODO if path starts with ':' then consider it as absolute path else as a relative path
  if( m_dataset_ptr != NULL )
  {
    yat::String path = keypath;
    
    std::vector<yat::String> keys;
    path.replace( "/", ":" );
    path.split( ':', &keys );
    
    LogicalGroupPtr tmp = m_dataset_ptr->getLogicalRoot();

    for( unsigned int i = 0; i < keys.size() && ! tmp.is_null(); i++ )
    {
       tmp = tmp->getGroup( KeyPtr( new Key( keys[i], Key::GROUP ) ) );
    }
    
    return tmp;
  }
  return NULL;
}

//-----------------------------------------------------------------------------
// LogicalGroup::getKeyNames
//-----------------------------------------------------------------------------
std::list<std::string> LogicalGroup::getKeyNames(IContainer::Type)
{
  THROW_NOT_IMPLEMENTED("LogicalGroup::getKeyNames");
}

//-----------------------------------------------------------------------------
// LogicalGroup::getKeys
//-----------------------------------------------------------------------------
std::list<KeyPtr> LogicalGroup::getKeys()
{
  CDMA_FUNCTION_TRACE("LogicalGroup::getKeys");
  std::list< KeyPtr > result;

  // Get children keys' names
  yat::String key = m_key_ptr.is_null() ? "" : m_key_ptr->getName();
  if( m_listkey_ptr.is_null() )
  {
    m_listkey_ptr = m_dictionary_ptr->getKeys( key );
  }
  for( StringList::iterator itChildren = m_listkey_ptr->begin(); itChildren != m_listkey_ptr->end(); itChildren++ )
  {
    result.push_back( KeyPtr(new Key( *itChildren, m_dictionary_ptr->getKeyType(*itChildren)) ) );
  }
  return result;
}

//-----------------------------------------------------------------------------
// LogicalGroup::bindKey
//-----------------------------------------------------------------------------
KeyPtr LogicalGroup::bindKey(const std::string&, const KeyPtr&)
{
  THROW_NOT_IMPLEMENTED("LogicalGroup::bindKey");
}

//-----------------------------------------------------------------------------
// LogicalGroup::setParent
//-----------------------------------------------------------------------------
void LogicalGroup::setParent(LogicalGroup*)
{
  THROW_NOT_IMPLEMENTED("LogicalGroup::setParent");
}

//-----------------------------------------------------------------------------
// LogicalGroup::getParent
//-----------------------------------------------------------------------------
LogicalGroupPtr LogicalGroup::getParent() const
{
  LogicalGroupPtr res (NULL);
  if( m_dataset_ptr != NULL )
  {
    LogicalGroupPtr tmp = m_dataset_ptr->getLogicalRoot();
    res = tmp->getGroup( m_parent_path );
  }
  return res;
}

//-----------------------------------------------------------------------------
// LogicalGroup::setParent
//-----------------------------------------------------------------------------
std::string LogicalGroup::getLocation() const
{
  yat::String path = m_parent_path;
  
  if( path != "" )
  {
    path += ":";
  }
  path += getName();

  return path;
}

//-----------------------------------------------------------------------------
// LogicalGroup::setParent
//-----------------------------------------------------------------------------
std::string LogicalGroup::getName() const
{
  return m_key_ptr.is_null() ? "" : m_key_ptr->getName();
}

//-----------------------------------------------------------------------------
// LogicalGroup::setParent
//-----------------------------------------------------------------------------
std::string LogicalGroup::getShortName() const
{
  return getName();
}

}
