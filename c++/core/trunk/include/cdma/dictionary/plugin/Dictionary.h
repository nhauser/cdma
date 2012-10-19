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

#ifndef __CDMA_DICTIONARY_H__
#define __CDMA_DICTIONARY_H__

#include <string>
#include <map>

// Include CDMA
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/impl/Key.h>

/// @cond dictAPI

namespace cdma
{


//==============================================================================
/// Keywords dictionary manager
//==============================================================================
class CDMA_DECL Dictionary 
{
friend class DataDefAnalyser;
friend class MapDefAnalyser;
friend class DictionaryConceptAnalyser;

public:
  class Concept
  {
  friend class DictionaryConceptAnalyser;
  friend class Dictionary;
  public:
    typedef std::map<std::string, std::string> AttributeMap;
    typedef std::list<std::string> SynonymList;

    bool isSynonym(const std::string &keyword);

    int id() const { return m_id; }
    const std::string& label() const { return m_label; }
    const std::string& description() const { return m_description; }
    const std::string& unit() const { return m_unit; }

  private:
    int         m_id;
    std::string m_label;           // the label is the primary keyword for the concept
    std::string m_description;     // concept explanation
    std::string m_unit;            // physical unit of the underlying data
    AttributeMap m_attribute_map;  // list of wished attributes
    SynonymList m_synonym_list;    // other keywords associated to this concept
  };
typedef yat::SharedPtr<Concept> ConceptPtr;

typedef std::multimap<std::string, std::string> KeyConnectionMap;
typedef std::map<int, SolverList> ConceptIdSolverListMap;
typedef std::map<int, ConceptPtr> IdConceptMap;
typedef std::map<std::string, int> KeywordConceptIdMap;

private:

  KeyConnectionMap     m_connection_map;    // connection between parent key and children keys
  ConceptIdSolverListMap m_concept_solvers_map;// Association between concept and solvers list
  KeywordConceptIdMap m_key_concept_map;   // Association between application keywords & concepts
  IdConceptMap        m_id_concept_map;    // The concepts
  std::string         m_key_file_name;     // Data definition document name
  std::string         m_spec_dict_name;    // Optional Specific dictionnary of concepts
  std::string         m_mapping_file_name; // Mapping document
  std::string         m_data_def_name;     // Data definition name
  std::string         m_mapping_name;      // Mapping name
  std::string         m_plugin_id;         // Plugin who created this object

  typedef std::multimap<std::string, std::string>::const_iterator connection_map_const_iterator;
  typedef std::pair<connection_map_const_iterator, connection_map_const_iterator> 
          connection_map_const_range;

  ConceptPtr createConcept(const std::string &label);
  ConceptPtr getConcept(const std::string &keyword);
  int getConceptId(const std::string &keyword);

public:

  /// c-tor
  Dictionary();
  Dictionary(const std::string &plugin_id);
  
  ~Dictionary();

  /// Get the version number (in 3 digits default implementation) that is plug-in
  /// dependent. This version corresponds of the dictionary defining the path. It  
  /// permits to distinguish various generation of IDataset for a same institutes.
  /// Moreover it's required to select the right class when using a IClassLoader
  /// invocation. 
  ///
  std::string getVersionNum();
    
  /// Read all keys stored in the XML dictionary file
  ///
  void readEntries() throw ( Exception );
  
  /// Return the path to reach the key dictionary file
  ///
  std::string& getKeyFileName() { return m_key_file_name; }
  
  /// Return the path to reach the mapping dictionary file
  ///
  std::string getMappingFileName() { return m_mapping_file_name; }
  
  /// Return the name of the data definition
  ///
  std::string getDataDefName() { return m_data_def_name; }
  
  /// Return the name of the mapping document dictionary
  ///
  std::string getMappingName() { return m_mapping_name; }
  
  /// Return the ordered list of the solvers associated with a key
  ///
  SolverList getSolversList(const IKeyPtr& key_ptr);
  
  /// Return all keys referenced in the dictionary.
  ///
  /// @return SharedPtr to a std::list of std::string objects
  ///
  StringListPtr getAllKeys();

  /// Return key names belonging to the given parent key
  ///
  /// @return SharedPtr to a std::list of std::string objects
  ///
  StringListPtr getKeys(const std::string& parent_key) throw( Exception );

  /// Return key type
  ///
  IKey::Type getKeyType(const std::string& key) throw( Exception );
  
  /// @param key
  ///            key object
  /// @return true or false
  ///
  bool containsKey(const std::string& key);

  /// Set the key file name
  /// The key file will be searched in $CDMA_DICTIONARY_PATH/views/
  ///
  void setKeyFileName(const std::string& file_name) { m_key_file_name = file_name; }
  
  /// Set the mapping file relative path and name
  /// typically: {plugin_name}/{mapping_file.xml}
  /// The mapping file will be searched in $CDMA_DICTIONARY_PATH/mappings/
  ///
  void setMappingFileName(const std::string& file_name) { m_mapping_file_name = file_name; }

  /// Set the (optionnal) concept dictionary file name
  /// The concepts file will be searched in $CDMA_DICTIONARY_PATH/concepts/
  ///
  void setSpecificConceptDictionary(const std::string& file_name) { m_spec_dict_name = file_name; }
};

DECLARE_SHARED_PTR(Dictionary);

} //namespace

/// @endcond clientAPI

#endif //__CDMA_DICTIONARY_H__

