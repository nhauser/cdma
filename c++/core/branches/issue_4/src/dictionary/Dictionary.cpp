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
#include <string>
#include <utility>
#include <yat/memory/SharedPtr.h>
#include <yat/utils/Logging.h>

#include <cdma/utils/SAXParsor.h>
#include <cdma/exception/impl/ExceptionImpl.h>
#include <cdma/dictionary/impl/Key.h>
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/factory/impl/FactoryImpl.h>

namespace cdma
{

const std::string s_core_dict_file = "core_concepts.xml";

//=============================================================================
// DictionaryConceptAnalyser
//
// Read dictionary type xml tree
//=============================================================================
class DictionaryConceptAnalyser: public SAXParsor::INodeAnalyser
{
private:
  Dictionary* m_dict_p;
  Dictionary::ConceptPtr m_current_concept_ptr;
  std::string m_last_attr_name;
  std::string m_last_content;

public:
  DictionaryConceptAnalyser(Dictionary* dict_p);

  INodeAnalyser* on_element(const yat::String& element_name, 
                            const SAXParsor::Attributes& attrs, 
                            const yat::String& current_file);

  void on_element_content(const yat::String&, const yat::String&, const yat::String&);

  void on_end_element(const yat::String& element_name);

  void release() { }
};

//----------------------------------------------------------------------------
// DictionaryConceptAnalyser::DictionaryConceptAnalyser
//----------------------------------------------------------------------------
DictionaryConceptAnalyser::DictionaryConceptAnalyser(Dictionary* dict_p)
: m_dict_p(dict_p)
{

}

//----------------------------------------------------------------------------
// DictionaryConceptAnalyser::on_element
//----------------------------------------------------------------------------
SAXParsor::INodeAnalyser* DictionaryConceptAnalyser::on_element(const yat::String& element_name, 
                                                      const SAXParsor::Attributes& attrs, 
                                                      const yat::String&)
{
  if( element_name.is_equal("dictionary") ||
      element_name.is_equal("definition") ||
      element_name.is_equal("attributes") ||
      element_name.is_equal("synonyms") ||
      element_name.is_equal("key") ||
      element_name.is_equal("unit") ||
      element_name.is_equal("description") )
  {
    // do nothing
  }
  
  else if( element_name.is_equal("concept") )
  {
    yat::String name;
    FIND_ATTR_VALUE(attrs, "label", name);
    m_current_concept_ptr = m_dict_p->createConcept(name);
  }

  else
  {
    THROW_EXCEPTION( "BAD_CONFIG",
                           PSZ_FMT( "Unknown element '%s' while parsing dicrtionary document",
                                      PSZ(element_name) ),
                           "DictionaryConceptAnalyser::on_element" );
  }
  
  return this;
}

//----------------------------------------------------------------------------
// DictionaryConceptAnalyser::on_element_content
//----------------------------------------------------------------------------
void DictionaryConceptAnalyser::on_element_content(const yat::String&, 
                                                   const yat::String& el_content, 
                                                   const yat::String&)
{
  m_last_content = el_content;
}

//----------------------------------------------------------------------------
// DictionaryConceptAnalyser::on_end_element
//----------------------------------------------------------------------------
void DictionaryConceptAnalyser::on_end_element(const yat::String& el_name)
{
  if( el_name.is_equal("key") )
    m_current_concept_ptr->m_synonym_list.push_back(m_last_content);

  else if( el_name.is_equal("description") )
    m_current_concept_ptr->m_description = m_last_content;

  else if( el_name.is_equal("unit") )
    m_current_concept_ptr->m_unit = m_last_content;

  else if( el_name.is_equal("attribute") )
    m_current_concept_ptr->m_attribute_map[m_last_attr_name] = m_last_content;
}

//=============================================================================
// DataDefAnalyser
//
// Read data-def type xml tree
//=============================================================================
class DataDefAnalyser: public SAXParsor::INodeAnalyser
{
private:
  Dictionary* m_dict_ptr;
  std::stack<std::string> m_current_group_name;
  
public:
  DataDefAnalyser(Dictionary* dict_ptr);

  INodeAnalyser* on_element(const yat::String& element_name, 
                            const SAXParsor::Attributes& attrs, 
                            const yat::String& current_file);

  void on_element_content(const yat::String&, const yat::String&, const yat::String&) {}

  void on_end_element(const yat::String& element_name);

  void release() { }
};

//----------------------------------------------------------------------------
// DataDefAnalyser::on_element
//----------------------------------------------------------------------------
DataDefAnalyser::DataDefAnalyser(Dictionary* dict_ptr) : m_dict_ptr(dict_ptr)
{
  m_current_group_name.push("root");
}

//----------------------------------------------------------------------------
// DataDefAnalyser::on_element
//----------------------------------------------------------------------------
SAXParsor::INodeAnalyser* DataDefAnalyser::on_element(const yat::String& element_name, 
                                                      const SAXParsor::Attributes& attrs, 
                                                      const yat::String&)
{
  if( element_name.is_equal("data-def") )
  {
    yat::String name;
    FIND_ATTR_VALUE(attrs, "name", name);
    m_dict_ptr->m_data_def_name = name;
  }
  
  else if( element_name.is_equal("group") )
  {
    yat::String key;
    FIND_ATTR_VALUE(attrs, "key", key);
    m_dict_ptr->m_connection_map.insert(std::pair<std::string,std::string>(m_current_group_name.top(), key));
    m_current_group_name.push(key);
  }
  
  else if( element_name.is_equal("item") )
  {
    yat::String key;
    FIND_ATTR_VALUE(attrs, "key", key);
    int concept_id = 0;
    try
    {
      concept_id = m_dict_ptr->getConceptId(key);
    }
    catch( ... )
    {
      // There is not defined concept related to this keyword, let's create it
      concept_id = m_dict_ptr->createConcept(key)->id();
      yat::log_notice( "dict", 
                       PSZ_FMT( "keyword '%s' doesn't refer to a defined concept", 
                                 PSZ(key) ) );
    }
    m_dict_ptr->m_key_concept_map[key] = concept_id;
    m_dict_ptr->m_connection_map.insert(std::pair<std::string,std::string>(m_current_group_name.top(), key));
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
    m_current_group_name.pop();
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
  Dictionary*  m_dict_ptr;
  int          m_current_concept_id;
  std::string  m_current_key;

  SolverList& get_solvers_list(int concept_id);
  void remove_from_solvers_list(int concept_id);

public:
  MapDefAnalyser(Dictionary* dict_ptr) : m_dict_ptr(dict_ptr)  {}

  INodeAnalyser* on_element(const yat::String& element_name, 
                            const SAXParsor::Attributes& attrs, 
                            const yat::String& current_file);

  void on_element_content(const yat::String& element_name, 
                          const yat::String& element_content, 
                          const yat::String& current_file);

  void on_end_element(const yat::String&) {}

  void release() { }
};

//----------------------------------------------------------------------------
// DataDefAnalyser::on_element
//----------------------------------------------------------------------------
SAXParsor::INodeAnalyser* MapDefAnalyser::on_element(const yat::String& element_name, 
                                                     const SAXParsor::Attributes& attrs, 
                                                     const yat::String&)
{
  if( element_name.is_equal("map-def") )
  {
    yat::String name;
    FIND_ATTR_VALUE(attrs, "name", name);
    m_dict_ptr->m_mapping_name = name;
  }

  if( element_name.is_equal("item") )
  {
    FIND_ATTR_VALUE_NO_THROW(attrs, "key", m_current_key);
    try
    {
      m_current_concept_id = m_dict_ptr->getConceptId(m_current_key);
    }
    catch( ... )
    {
      // The key may be not used by the application and may
      // not refer to a fully defined concept
      m_current_concept_id = 0;
      yat::log_notice( "dict", 
                       PSZ_FMT( "key '%s' not defined in data definition document", 
                                 PSZ(m_current_key) ) );
    }
  }
  return this;
}

//----------------------------------------------------------------------------
// MapDefAnalyser::get_solvers_list
//----------------------------------------------------------------------------
SolverList& MapDefAnalyser::get_solvers_list(int concept_id)
{
  Dictionary::ConceptIdSolverListMap::iterator it = m_dict_ptr->m_concept_solvers_map.find(concept_id);
  if( it == m_dict_ptr->m_concept_solvers_map.end() )
  {
    // Creates the solver list
    m_dict_ptr->m_concept_solvers_map[concept_id] = SolverList();
  }
  return m_dict_ptr->m_concept_solvers_map[concept_id];
}

//----------------------------------------------------------------------------
// MapDefAnalyser::remove_from_solvers_list
//----------------------------------------------------------------------------
void MapDefAnalyser::remove_from_solvers_list(int concept_id)
{
  Dictionary::ConceptIdSolverListMap::iterator it = m_dict_ptr->m_concept_solvers_map.find(concept_id);
  if( it != m_dict_ptr->m_concept_solvers_map.end() )
  {
    m_dict_ptr->m_concept_solvers_map.erase(it);
  }
}

//----------------------------------------------------------------------------
// MapDefAnalyser::on_element_content
//----------------------------------------------------------------------------
void MapDefAnalyser::on_element_content(const yat::String& element_name, 
                                        const yat::String& element_content, 
                                        const yat::String& /*current_file*/)
{
  if( m_current_concept_id && element_name.is_equal("path") )
  {
    KeyPathPtr path_ptr = new KeyPath(element_content);

    // Push the KeyPath object at the back of the list
    get_solvers_list(m_current_concept_id).push_back( IKeySolverPtr(path_ptr) );
  }
  else if( m_current_concept_id && element_name.is_equal("call") )
  {
    IPluginMethodPtr method_ptr = FactoryImpl::getPluginMethod(m_dict_ptr->m_plugin_id, element_content);
    if( method_ptr )
    {
      KeyMethodPtr key_method_ptr = new KeyMethod(element_content, method_ptr);

      // Push the KeyMethod object reference at the back of the list
      get_solvers_list(m_current_concept_id).push_back( IKeySolverPtr(key_method_ptr) );
    }
    else
    {
      yat::log_error( "dict", PSZ_FMT("unable to solve key '%s'. method '%s' not implemented by plugin",
                      PSZ(m_current_key), PSZ(element_content) ) );
    }
  }
}

//----------------------------------------------------------------------------
// Dictionary::Dictionary
//----------------------------------------------------------------------------
Dictionary::Dictionary(const std::string &plugin_id) : m_plugin_id(plugin_id)
{
  m_key_file_name = Factory::getActiveView() + "_view.xml";
}
    
//----------------------------------------------------------------------------
// Dictionary::Dictionary
//----------------------------------------------------------------------------
Dictionary::Dictionary()
{
  m_key_file_name = Factory::getActiveView() + "_view.xml";
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
// Dictionary::readEntries
//----------------------------------------------------------------------------
void Dictionary::readEntries() throw ( Exception )
{
  try
  {
    CDMA_TRACE("reading concepts...");

    // Read the main (core) dictionary of concepts
    DictionaryConceptAnalyser core_dict_analyser(this);
    SAXParsor::start(Factory::getConceptDictionaryFolder() + s_core_dict_file, &core_dict_analyser);

    if( !m_spec_dict_name.empty() )
    {
      CDMA_TRACE("reading specific concepts: " << m_spec_dict_name);
      // Read a more specific dictionary of concepts
      DictionaryConceptAnalyser spec_dict_analyser(this);
      SAXParsor::start(Factory::getConceptDictionaryFolder() + m_spec_dict_name, &spec_dict_analyser);
    }

    CDMA_TRACE("reading data definition");
    // Read the data definition expected by the client app
    DataDefAnalyser datadef_analyser(this);
    SAXParsor::start(Factory::getKeyDictionaryFolder() + m_key_file_name, &datadef_analyser);
    
    CDMA_TRACE("reading keywords mapping");
    // Read the mapping document
    MapDefAnalyser mapdef_analyser(this);
    SAXParsor::start(Factory::getMappingDictionaryFolder(m_plugin_id) + m_mapping_file_name,
                     &mapdef_analyser);
  }
  catch( yat::Exception& e )
  {
    e.push_error("READ_ERROR", "Cannot read dictionary documents", 
                 "Dictionary::readEntries");
    RE_THROW_EXCEPTION(e);
  }
}

//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
StringListPtr Dictionary::getAllKeys()
{
  StringList* pList = new StringList;
  for(KeywordConceptIdMap::const_iterator cit = m_key_concept_map.begin();
      cit !=  m_key_concept_map.end(); cit++)
  {
    pList->push_back(cit->first);
  }
  return pList;
}

//----------------------------------------------------------------------------
// Dictionary::getKeys
//----------------------------------------------------------------------------
StringListPtr Dictionary::getKeys(const std::string& parent_key) throw( Exception )
{
  CDMA_FUNCTION_TRACE("Dictionary::getKeys");
  connection_map_const_range range = m_connection_map.equal_range(parent_key);
  StringList* pList = new StringList;
  for(connection_map_const_iterator cit = range.first; cit != range.second; cit++)
  {    
    pList->push_back(cit->second);
  }
  return pList;
}

//----------------------------------------------------------------------------
// Dictionary::getKeyType
//----------------------------------------------------------------------------
Key::Type Dictionary::getKeyType(const std::string& key) throw( Exception )
{
  KeywordConceptIdMap::const_iterator citKey = m_key_concept_map.find(key);
  if( citKey == m_key_concept_map.end() )
    THROW_EXCEPTION( "KEY_NOT_FOUND", 
                           PSZ_FMT( "Key '%s' not found in data definition", 
                                    PSZ(key) ), "Dictionary::getPath" );

  // Looking for group key
  if( m_connection_map.find(key) != m_connection_map.end() )
    return Key::GROUP;
    
  return Key::ITEM;
}

//----------------------------------------------------------------------------
// Dictionary::getSolversList
//----------------------------------------------------------------------------
SolverList Dictionary::getSolversList(const IKeyPtr& key_ptr)
{
  std::string key = key_ptr->getName();

  // Retreive the key id
  KeywordConceptIdMap::const_iterator citKey = m_key_concept_map.find(key);
  if( citKey == m_key_concept_map.end() )
    THROW_EXCEPTION( "KEY_NOT_FOUND", 
                           PSZ_FMT( "Key '%s' not found in data definition", 
                                    PSZ(key) ), "Dictionary::getSolversList" );
  int concept_id = citKey->second;

  ConceptIdSolverListMap::iterator itSolvers = m_concept_solvers_map.find(concept_id);
  if( itSolvers == m_concept_solvers_map.end() )
    THROW_EXCEPTION( "NO_DATA", 
                           PSZ_FMT( "Key '%s' isn't associated with any solver", 
                                    PSZ(key) ), "Dictionary::getSolversList" );

  return itSolvers->second;
}

//----------------------------------------------------------------------------
// Dictionary::getDictionary
//----------------------------------------------------------------------------
bool Dictionary::containsKey( const std::string& )
{
  THROW_NOT_IMPLEMENTED("Dictionary::containsKey");
}

//----------------------------------------------------------------------------
// Dictionary::createConcept
//----------------------------------------------------------------------------
Dictionary::ConceptPtr Dictionary::createConcept(const std::string &label)
{
  static int s_concept_id = 0;

  ConceptPtr concept_ptr = new Dictionary::Concept;
  concept_ptr->m_label = label;
  concept_ptr->m_id = ++s_concept_id;
  m_id_concept_map[s_concept_id] = concept_ptr;
  return concept_ptr;
}

//----------------------------------------------------------------------------
// Dictionary::getConcept
//----------------------------------------------------------------------------
Dictionary::ConceptPtr Dictionary::getConcept(const std::string &keyword)
{
  for( IdConceptMap::const_iterator cit = m_id_concept_map.begin();
       cit != m_id_concept_map.end(); ++cit )
  {
    if( cit->second->isSynonym(keyword) )
      return cit->second;
  }
  THROW_EXCEPTION("NO_DATA", 
                       PSZ_FMT("No concept for this keyword: %s", PSZ(keyword)),
                       "Dictionary::getConcept");
}

//----------------------------------------------------------------------------
// Dictionary::getConceptId
//----------------------------------------------------------------------------
int Dictionary::getConceptId(const std::string &keyword)
{
  for( IdConceptMap::const_iterator cit = m_id_concept_map.begin();
       cit != m_id_concept_map.end(); ++cit )
  {
    if( cit->second->isSynonym(keyword) )
      return cit->first;
  }
  THROW_EXCEPTION("NO_DATA", 
                       PSZ_FMT("No concept for this keyword: %s", PSZ(keyword)),
                       "Dictionary::getConcept");
}

//----------------------------------------------------------------------------
// Dictionary::Concept::isSynonym
//----------------------------------------------------------------------------
bool Dictionary::Concept::isSynonym( const std::string &keyword )
{
  if( keyword == m_label )
    return true;

  for( SynonymList::const_iterator cit = m_synonym_list.begin(); 
       cit != m_synonym_list.end(); ++cit )
  {
    if( keyword == *cit )
      return true;
  }

  return false;
}

} //namespace



