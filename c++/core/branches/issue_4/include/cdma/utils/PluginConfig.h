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

#ifndef __CDMA_PLUGIN_CONFIGURATION_H__
#define __CDMA_PLUGIN_CONFIGURATION_H__

// STD
#include <string>
#include <map>

// YAT
#include <yat/utils/Singleton.h>
#include <yat/any/Any.h>

// CDMA
#include <cdma/Common.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/IKey.h>

namespace cdma
{

//==============================================================================
/// DatasetModel
/// Configuration description relative to a specific Plug-in
/// This class is aimed to help
/// - the mapping file detection
/// - defining specific parameter to be applied to the plug-in/engine when 
///   retrieving data for a specific kind of datasets
//==============================================================================
class DatasetModel
{
public:

  typedef std::map<std::string, std::string> ParamMap;

  //==============================================================================
  /// Criterion
  /// A information that will help dataset recognition in order to choose
  /// the correct mapping file
  //==============================================================================
  class Criterion
  {
  public:

    enum Type
    {
      EXIST = 0,
      EQUAL,
      NOT_EQUAL
    };

    /// c-tor
    Criterion(const std::string& target_path, Type type, const yat::Any& value)
        :  m_target_path(target_path), m_type(type), m_criterion_value(value)
        {    }

    /// Test the criterion on the given data set
    ///
    /// @return true on succeed
    ///
    bool doTest(IDataset* dataset_p) const;

  private:
    /// target path to test
    std::string m_target_path;

    /// Criterion type
    Type m_type;

    /// Criterion value, according to its type
    yat::Any m_criterion_value;
  };

  //==============================================================================
  /// DatasetParameter
  /// Parameter rerieved in the dataset
  //==============================================================================
  class DatasetParameter
  {
  public:

    /// type of action to get parameter value
    enum Type
    {
      NAME = 0,
      VALUE,
      EXIST,
      EQUAL,
      CONSTANT
    };

    /// c-tor
    DatasetParameter()
    {  }

    DatasetParameter(const std::string& target_path, Type type, bool b, const std::string& value)
      : m_target_path(target_path), m_type(type), 
        m_test_result(b), m_value(value)
    {  }

    DatasetParameter(const DatasetParameter& other)
      : m_target_path(other.m_target_path), m_type(other.m_type), 
        m_test_result(other.m_test_result), m_value(other.m_value)
    {  }

    /// Retrieving value as a string
   std::string getValue(IDataset* dataset_p) const;

  private:

    /// target path to read to extract parameter
    std::string m_target_path;

    /// type of action to get parameter value
    Type m_type;

    /// expected result for test actions (EXIST/EQUAL) 
    bool m_test_result;

    // for EQUAL    type: comparaison value
    // for CONSTANT type: actual value
    std::string m_value;
  };

  /// c-tor
  DatasetModel(const std::string &name) : m_name(name) { }

  /// Adding criterion
  void addCriterion(const std::string& target_path, Criterion::Type type, 
                    const yat::Any& criterion_value);

  /// Adding dataset parameter
  void addDatasetParameter(const std::string& name, DatasetParameter::Type type, bool test_result, 
                           const std::string& target_path, const std::string& value);

  /// Add plugin parameter
  void addPluginParameter(const std::string& name, const std::string& value);

  /// Applying criteria
  ///
  /// @return true if the dataset match the criteria
  ///
  bool applyCriteria(IDataset* dataset_p) const;

  /// Get all parameters (dataset & plugin parameters)
  ///
  /// @return a key/value map 
  ///
  void getAllParameters(IDataset* dataset_p, DatasetModel::ParamMap* params_p) const;

  /// Retrun the model's name
  ///
  const std::string& getName() const { return m_name; }

private:
  std::string m_name;
  std::list<Criterion> m_list_criteria;
  std::map<std::string, DatasetParameter> m_map_dataset_parameter;
  ParamMap m_map_plugin_parameter;

  std::string getValue(IDataset* dataset_p, const std::string& name) const;
};

DECLARE_CLASS_SHARED_PTR(DatasetModel);
typedef std::list<DatasetModelPtr> DatasetModelList;
typedef std::map<std::string, DatasetModelList> DatasetModelMap;

//==============================================================================
/// @brief Plugins configuration manager
///
//==============================================================================
class CDMA_DECL PluginConfigManager : public yat::Singleton<PluginConfigManager>
{
friend class PluginConfigAnalyser;

public:

  /// Load a configuration file for a dedicated plugins
  ///
  /// @param plugin_id plugin identifier
  /// @param file_name configuration file name
  ///
  /// @throw cdma::Exception if an error occur
  ///
  static void load(const std::string& plugin_id, const std::string& file_name) 
                                                               throw ( cdma::Exception );

  /// Ask for the configuration of a plugin matching a dataset
  ///
  /// @param plugin_id plugin identifier
  /// @param dataset_ptr the dataset on which to apply the configuration definition
  ///
  /// @return a key-value map of parameters defining a dataset
  ///
  /// @throw an exception if the configuration don't exist
  ///
  static void getConfiguration(const std::string& plugin_id,
                                      IDataset* dataset_p, 
                                      DatasetModel::ParamMap* map_param) throw ( cdma::Exception );


private:
  DatasetModelMap m_map_model;
};


}

#endif
