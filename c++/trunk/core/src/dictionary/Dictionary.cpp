//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************

#include <string>
#include <utility>
#include <yat/memory/SharedPtr.h>
#include <yat/utils/Logging.h>

#include <cdma/utils/SAXParsor.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/Dictionary.h>
#include <cdma/dictionary/Path.h>

namespace cdma
{

//=============================================================================
// DataDefAnalyser
//
// Read data-def type xml tree
//=============================================================================
class DataDefAnalyser: public SAXParsor::INodeAnalyser
{
private:
  Dictionary* m_pDict;
  std::stack<int> m_current_group_id;
  
public:
  DataDefAnalyser(Dictionary* pDict);
  INodeAnalyser* on_element(const yat::String& element_name, const SAXParsor::Attributes& attrs, const yat::String& current_file);
  void on_element_content(const yat::String&, const yat::String&, const yat::String&) {}
  void on_end_element(const yat::String& element_name);
  void release() { delete this; }
};

//----------------------------------------------------------------------------
// DataDefAnalyser::on_element
//----------------------------------------------------------------------------
DataDefAnalyser::DataDefAnalyser(Dictionary* pDict) : m_pDict(pDict)
{
  m_current_group_id.push(0);
}

//----------------------------------------------------------------------------
// DataDefAnalyser::on_element
//----------------------------------------------------------------------------
SAXParsor::INodeAnalyser* DataDefAnalyser::on_element(const yat::String& element_name, const SAXParsor::Attributes& attrs, const yat::String&)
{
  static int s_key_id = 0;
  
  if( element_name.is_equal("data-def") )
  {
    yat::String name;
    FIND_ATTR_VALUE(attrs, "name", name);
    m_pDict->m_data_def_name = name;
  }
  
  else if( element_name.is_equal("group") )
  {
    yat::String key;
    FIND_ATTR_VALUE(attrs, "key", key);
    m_pDict->m_key_id_map[key] = ++s_key_id;
    std::cout<<"Group: "<<key<<" Id: "<<s_key_id<< " Parent: "<<m_current_group_id.top()<<std::endl;
    m_pDict->m_connection_map.insert(std::pair<int,int>(m_current_group_id.top(), s_key_id));
    m_current_group_id.push(s_key_id);
  }
  
  else if( element_name.is_equal("item") )
  {
    yat::String key;
    FIND_ATTR_VALUE(attrs, "key", key);
    m_pDict->m_key_id_map[key] = ++s_key_id;
    std::cout<<"Item: "<<key<<" Id: "<<s_key_id<< " Parent: "<<m_current_group_id.top()<<std::endl;
    m_pDict->m_connection_map.insert(std::pair<int,int>(m_current_group_id.top(), s_key_id));
  }
  
  return this;
}

//----------------------------------------------------------------------------
// DataDefAnalyser::on_end_element
//----------------------------------------------------------------------------
void DataDefAnalyser::on_end_element(const yat::String& element_name)
{
  if( element_name.is_equal("group") )
  {
    m_current_group_id.pop();
  }
}

//=============================================================================
// MapDefAnalyser
//
// Read map-def type xml tree
//=============================================================================
class MapDefAnalyser: public SAXParsor::INodeAnalyser
{
private:
  Dictionary* m_pDict;
  std::string  m_current_key;
  
public:
  MapDefAnalyser(Dictionary* pDict) : m_pDict(pDict)  {}
  INodeAnalyser* on_element(const yat::String& element_name, const SAXParsor::Attributes& attrs, const yat::String& current_file);
  void on_element_content(const yat::String& element_name, const yat::String& element_content, const yat::String& current_file);
  void on_end_element(const yat::String&) {}
  void release() { delete this; }
};

//----------------------------------------------------------------------------
// DataDefAnalyser::on_element
//----------------------------------------------------------------------------
SAXParsor::INodeAnalyser* MapDefAnalyser::on_element(const yat::String& element_name, const SAXParsor::Attributes& attrs, const yat::String&)
{  
  if( element_name.is_equal("map-def") )
  {
    yat::String name;
    FIND_ATTR_VALUE(attrs, "name", name);
    m_pDict->m_mapping_name = name;
  }
  
  if( element_name.is_equal("item") )
  {
    FIND_ATTR_VALUE_NO_THROW(attrs, "key", m_current_key);
    if( m_pDict->m_key_id_map.find(m_current_key) == m_pDict->m_key_id_map.end() )
      yat::log_warning("dict", PSZ_FMT("key '%s' not defined in data definition document", PSZ(m_current_key)));
  }
  return this;
}

//----------------------------------------------------------------------------
// MapDefAnalyser::on_element_content
//----------------------------------------------------------------------------
void MapDefAnalyser::on_element_content(const yat::String& element_name, const yat::String& element_content, const yat::String& current_file)
{
  if( element_name.is_equal("path") )
  {
    if( !m_current_key.empty() )
    {
      PathPtr ptrPath = new Path();
      ptrPath->setValue(element_content);
      m_pDict->m_key_path_map[m_pDict->m_key_id_map[m_current_key]] = ptrPath;
    }
  }
}

//----------------------------------------------------------------------------
// Dictionary::Dictionary
//----------------------------------------------------------------------------
Dictionary::Dictionary()
{
}
    
//----------------------------------------------------------------------------
// Dictionary::~Dictionary
//----------------------------------------------------------------------------
Dictionary::~Dictionary()
{
}
    
//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
std::string Dictionary::getVersionNum()
{
  THROW_NOT_IMPLEMENTED("Dictionary::getVersionNum");
}

//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
void Dictionary::readEntries() throw ( Exception )
{
  SAXParsor parsor;
  
  try
  {
    DataDefAnalyser datadef_analyser(this);
    parsor.start(m_key_file_path, &datadef_analyser);
    
    MapDefAnalyser mapdef_analyser(this);
    parsor.start(m_mapping_file_path, &mapdef_analyser);
  }
  catch( yat::Exception& e )
  {
    e.push_error("READ_ERROR", "Cannot read dictionary documents", "Dictionary::readEntries");
    throw Exception(e);
  }
}

//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
StringListPtr Dictionary::getAllKeys()
{
  StringList* pList = new StringList;
  for(std::map<std::string, int>::const_iterator cit = m_key_id_map.begin();
      cit !=  m_key_id_map.end(); cit++)
  {
    pList->push_back(cit->first);
  }
  return pList;
}

//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
StringListPtr Dictionary::getKeys(const std::string& parent_key) throw( Exception )
{
  int parent_id = 0;
  if( !parent_key.empty() )
  {
    std::map<std::string, int>::const_iterator citKey = m_key_id_map.find(parent_key);
    if( citKey == m_key_id_map.end() )
      throw cdma::Exception("KEY_NOT_FOUND",
                            yat::String::str_format("Group key '%s' not found in data definition",PSZ(parent_key)),
                            "Dictionary::getKeys");
    parent_id = citKey->second;
  }
  
  // Build the reverse of the map key
  std::map<int, std::string> reverse_key_map;
  for(std::map<std::string, int>::const_iterator cit = m_key_id_map.begin();
      cit !=  m_key_id_map.end(); cit++)
  {
    reverse_key_map[cit->second] = cit->first;
  }
  
  // Get the key ids the keys whose parent is the parent_key id
  // then use the above reverse map to retreive corresponding key names
  connection_map_const_range range = m_connection_map.equal_range(parent_id);
  StringList* pList = new StringList;
  for(connection_map_const_iterator cit = range.first; cit != range.second; cit++)
  {    
    pList->push_back(reverse_key_map[cit->second]);
  }
  return pList;
}

//----------------------------------------------------------------------------
// Dictionary::getKeyType
//----------------------------------------------------------------------------
Key::Type Dictionary::getKeyType(const std::string& key) throw( Exception )
{
  std::map<std::string, int>::const_iterator citKey = m_key_id_map.find(key);
  if( citKey == m_key_id_map.end() )
    throw cdma::Exception("KEY_NOT_FOUND", 
                          yat::String::str_format("Key '%s' not found in data definition",PSZ(key)),
                          "Dictionary::getPath");

  // Looking for group key
  if( m_connection_map.find(citKey->second) != m_connection_map.end() )
    return Key::Group;
    
  return Key::Item;
}

//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
PathPtr Dictionary::getPath(const KeyPtr& ptrKey) throw( Exception )
{
  std::string key = ptrKey->getName();
  std::map<std::string, int>::const_iterator citKey = m_key_id_map.find(key);
  if( citKey == m_key_id_map.end() )
    throw cdma::Exception("KEY_NOT_FOUND", 
                          yat::String::str_format("Key '%s' not found in data definition",PSZ(key)),
                          "Dictionary::getPath");
  
  int key_id = citKey->second;
  std::map<int, PathPtr>::const_iterator citPath = m_key_path_map.find(key_id);
  if( citPath == m_key_path_map.end() )
    throw cdma::Exception("NO_DATA", 
                          yat::String::str_format("No path for key '%s' in mapping dictionary",PSZ(key)),
                          "Dictionary::getPath");
  
  return citPath->second;
}

//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
std::list<PathPtr> Dictionary::getAllPaths(const KeyPtr& key)
{
  THROW_NOT_IMPLEMENTED("Dictionary::getAllPaths");
}

//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
bool Dictionary::containsKey(const std::string& key)
{
  THROW_NOT_IMPLEMENTED("Dictionary::containsKey");
}

} //namespace



