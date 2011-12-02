//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_DICTIONARY_H__
#define __CDMA_DICTIONARY_H__

#include <string>
#include <map>
#include <yat/memory/SharedPtr.h>

// Include CDMA
#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/Key.h>

namespace cdma
{

//=============================================================================
/// Dictionary
///
/// 
//=============================================================================
class Dictionary 
{
friend class DataDefAnalyser;
friend class MapDefAnalyser;
  
private:

  std::multimap<int, int>              m_connection_map;    // connection between parent (key) and children (data)
  std::map<std::string, int>           m_key_id_map;        // the keys and the associated identifier
  std::map<int, PathPtr>               m_key_path_map;      // Association between key index and path
  std::multimap<int, std::string>      m_key_synonyms_map;  // Association between key index and synonyms

  std::string                          m_key_file_path;     // Path to the data definition document
  std::string                          m_mapping_file_path; // Path to the mapping document (the dict itself)
  std::string                          m_data_def_name;     // Data definition name
  std::string                          m_mapping_name;      // Mapping name
  
  typedef std::multimap<int, int>::const_iterator connection_map_const_iterator;
  typedef std::pair<connection_map_const_iterator, connection_map_const_iterator> connection_map_const_range;

public:

  /// c-tor
  //Dictionary(const std::string& datadef_path, const std::string& mapdef_path);
  Dictionary();
  
  ~Dictionary();

  /// Get the version number (in 3 digits default implementation) that is plug-in
  /// dependent. This version corresponds of the dictionary defining the path. It  
  /// permits to distinguish various generation of IDataset for a same institutes.
  /// Moreover it's required to select the right class when using a IClassLoader
  /// invocation. 
  ///
  std::string getVersionNum();
  
  /// Get the plug-in implementation of a IClassLoader so invocations of external
  /// are made possible.
  ///
  //## yat::SharedPtr<IClassLoader, yat::Mutex> getClassLoader();
  
  /// Read all keys stored in the XML dictionary file
  ///
  void readEntries() throw ( Exception );
  
  /// Return the path to reach the key dictionary file
  ///
  std::string& getKeyFilePath() { return m_key_file_path; }
  
  /// Return the path to reach the mapping dictionary file
  ///
  std::string getMappingFilePath() { return m_mapping_file_path; }
  
  /// Return the name of the data definition
  ///
  std::string getDataDefName() { return m_data_def_name; }
  
  /// Return the name of the mapping document dictionary
  ///
  std::string getMappingName() { return m_mapping_name; }
  
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
  Key::Type getKeyType(const std::string& key) throw( Exception );
  
  /// Get the path referenced by the key. If there are more than one paths are
  /// referenced by the path, get the default one.
  ///
  /// @param key
  ///            key object
  /// @return std::string object
  ///
  PathPtr getPath(const KeyPtr& key) throw( Exception );

  /// Return all paths referenced by the key.
  ///
  /// @param key
  ///            key object
  /// @return a std::list of std::string objects
  ///
  std::list<PathPtr> getAllPaths(const KeyPtr& key);

  /// @param key
  ///            key object
  /// @return true or false
  ///
  bool containsKey(const std::string& key);

  /// Return the path to reach the key dictionary file
  ///
  void setKeyFilePath(const std::string& file_path) { m_key_file_path = file_path; }
  
  /// Return the path to reach the mapping dictionary file
  ///
  void setMappingFilePath(const std::string& file_path) { m_mapping_file_path = file_path; }
};

} //namespace

#endif //__CDMA_DICTIONARY_H__

