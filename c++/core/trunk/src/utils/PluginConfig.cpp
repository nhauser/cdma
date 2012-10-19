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

#include <yat/utils/Logging.h>

#include <cdma/utils/SAXParsor.h>
#include <cdma/exception/impl/ExceptionImpl.h>
#include <cdma/utils/PluginConfig.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/factory/Factory.h>

namespace cdma
{

//=============================================================================
// PluginConfigAnalyser
//
// Read configuration type xml tree
//=============================================================================
class PluginConfigAnalyser: public SAXParsor::INodeAnalyser
{
friend class PluginConfigManager;
private:
  PluginConfigManager* m_config_manager_p;
  DatasetModelPtr      m_current_dataset_model_ptr;
  std::string          m_current_plugin_id;

  void criterionAnalysis(const SAXParsor::Attributes& attrs);
  void parameterAnalysis(const SAXParsor::Attributes& attrs);
  void pluginParameterAnalysis(const SAXParsor::Attributes& attrs);

public:
  PluginConfigAnalyser(PluginConfigManager* p, const std::string plugin_id)
    : m_config_manager_p(p), m_current_plugin_id(plugin_id) { }

  INodeAnalyser* on_element(const yat::String& element_name, 
                            const SAXParsor::Attributes& attrs, 
                            const yat::String& current_file);
  void on_element_content(const yat::String&, const yat::String&, const yat::String&) {}
  void on_end_element(const yat::String& element_name);
  void release() { }
};

//----------------------------------------------------------------------------
// PluginConfigAnalyser::on_element
//----------------------------------------------------------------------------
SAXParsor::INodeAnalyser* PluginConfigAnalyser::on_element(const yat::String& element_name, 
                                                      const SAXParsor::Attributes& attrs, 
                                                      const yat::String&)
{
  if( element_name.is_equal_no_case("dataset-model") )
  {
    yat::String name;
    FIND_ATTR_VALUE(attrs, "name", name);
    m_current_dataset_model_ptr = new DatasetModel(name);
    return this;
  }
  
  else if( element_name.is_equal_no_case("plugin-config") ||
           element_name.is_equal_no_case("criteria") ||
           element_name.is_equal_no_case("parameters") ||
           element_name.is_equal_no_case("cpp") ||
           element_name.is_equal_no_case("plugin") )
  {
    return this;
  }

  else if( element_name.is_equal_no_case("java") || element_name.is_equal_no_case("global") )
  {
    // don't parse 'java' or 'global' section
    return NULL;
  }

  else if( element_name.is_equal_no_case("if") )
  {
    criterionAnalysis(attrs);
    return this;
  }
  
  else if( element_name.is_equal_no_case("parameter") )
  {
    parameterAnalysis(attrs);
    return this;
  }
  
  else if( element_name.is_equal_no_case("set") )
  {
    pluginParameterAnalysis(attrs);
    return this;
  }
  
  else
  {
    THROW_EXCEPTION(
      "BAD_CONFIG", 
      PSZ_FMT("Element '%s' in plugin configuration file is unknown", PSZ(element_name)),
      "PluginConfigAnalyser::on_element");
  }
}

//----------------------------------------------------------------------------
// PluginConfigAnalyser::criterionAnalysis
//----------------------------------------------------------------------------
void PluginConfigAnalyser::criterionAnalysis(const SAXParsor::Attributes& attrs)
{
  ///  <!ELEMENT if EMPTY>
  ///  <!ATTLIST if 
  ///      target CDATA #REQUIRED
  ///      exist ( true | false ) #IMPLIED
  ///      equal CDATA #IMPLIED
  ///      not_equal CDATA #IMPLIED
  ///  >

  yat::String target = SAXParsor::attribute_value(attrs, "target");
  DatasetModel::Criterion::Type type;
  yat::Any comp_value;

  if( SAXParsor::has_attribute(attrs, "exist") )
  {
    type = DatasetModel::Criterion::EXIST;
    yat::String test =  SAXParsor::attribute_value(attrs, "exist");
    if( test.is_equal_no_case( "true" ) || test.is_equal( "1" ) )
      comp_value = true;
    else
      comp_value = false;
  }
  else if( SAXParsor::has_attribute(attrs, "equal") )
  {
    type = DatasetModel::Criterion::EQUAL;
    comp_value =  SAXParsor::attribute_value(attrs, "equal");
  }
  else if( SAXParsor::has_attribute(attrs, "not_equal") )
  {
    type = DatasetModel::Criterion::NOT_EQUAL;
    comp_value =  SAXParsor::attribute_value(attrs, "not_equal");
  }
  else
  {
    THROW_EXCEPTION(
      "BAD_CONFIG", 
      PSZ_FMT("Criterion type is missing"),
      "PluginConfigAnalyser::criterionAnalysis");
  }

  m_current_dataset_model_ptr->addCriterion(target, type, comp_value);
}

//----------------------------------------------------------------------------
// PluginConfigAnalyser::parameterAnalysis
//----------------------------------------------------------------------------
void PluginConfigAnalyser::parameterAnalysis(const SAXParsor::Attributes& attrs)
{
  ///  <!ELEMENT parameter EMPTY>
  ///  <!ATTLIST parameter
  ///      name CDATA #REQUIRED
  ///      type ( exist | name | value | constant | equal ) #REQUIRED
  ///      test ( true | false ) #IMPLIED
  ///      target CDATA #IMPLIED
  ///      constant CDATA #IMPLIED
  ///  >

  yat::String name = SAXParsor::attribute_value(attrs, "name");
  yat::String type_value = SAXParsor::attribute_value(attrs, "type");
  yat::String target_value;
  DatasetModel::DatasetParameter::Type type;
  std::string value;
  bool test_result = true;

  std::map<std::string, DatasetModel::DatasetParameter::Type> types;
  types["name"] = DatasetModel::DatasetParameter::NAME;
  types["value"] = DatasetModel::DatasetParameter::VALUE;
  types["exist"] = DatasetModel::DatasetParameter::EXIST;
  types["equal"] = DatasetModel::DatasetParameter::EQUAL;
  types["constant"] = DatasetModel::DatasetParameter::CONSTANT;
  type = types[type_value];

  if( SAXParsor::has_attribute(attrs, "target") )
  {
    target_value = SAXParsor::attribute_value(attrs, "target");
  }

  if( SAXParsor::has_attribute(attrs, "test") )
  {
    yat::String test =  SAXParsor::attribute_value(attrs, "test");
    if( test.is_equal_no_case( "true" ) || test.is_equal( "1" ) )
      test_result = true;
    else
      test_result = false;
  }

  if( SAXParsor::has_attribute(attrs, "constant") )
  {
    value =  SAXParsor::attribute_value(attrs, "constant");
  }

  m_current_dataset_model_ptr->addDatasetParameter(name, type, test_result, target_value, value);
}

//----------------------------------------------------------------------------
// PluginConfigAnalyser::parameterAnalysis
//----------------------------------------------------------------------------
void PluginConfigAnalyser::pluginParameterAnalysis(const SAXParsor::Attributes& attrs)
{
  /// <!ELEMENT set EMPTY>
  /// <!ATTLIST set 
  ///     name CDATA #REQUIRED
  ///     value CDATA #REQUIRED
  /// >

  std::string name = SAXParsor::attribute_value(attrs, "name");
  std::string value = SAXParsor::attribute_value(attrs, "value");

  m_current_dataset_model_ptr->addPluginParameter(name, value);
}

//----------------------------------------------------------------------------
// DataDefAnalyser::on_end_element
//----------------------------------------------------------------------------
void PluginConfigAnalyser::on_end_element(const yat::String& element_name)
{
  if( element_name.is_equal("dataset-model") )
  {
    PluginConfigManager::instance().m_map_model[m_current_plugin_id].push_back(m_current_dataset_model_ptr);
  }
}

//==============================================================================
// DatasetModel
//==============================================================================
//----------------------------------------------------------------------------
// DatasetModel::addCriterion
//----------------------------------------------------------------------------
void DatasetModel::addCriterion(const std::string& target_path, Criterion::Type type, 
                           const yat::Any& value)
{
  m_list_criteria.push_back( Criterion(target_path, type, value) );
}

//----------------------------------------------------------------------------
// DatasetModel::addDatasetParameter
//----------------------------------------------------------------------------
void DatasetModel::addDatasetParameter(const std::string& name, DatasetParameter::Type type,
                                       bool test_result, const std::string& target_path, 
                                       const std::string& value)
{
  m_map_dataset_parameter[name] = DatasetParameter(target_path, type, test_result, value);
}

//----------------------------------------------------------------------------
// DatasetModel::addPluginParameter
//----------------------------------------------------------------------------
void DatasetModel::addPluginParameter(const std::string& name, const std::string& value)
{
  m_map_plugin_parameter[name] = value;
}

//----------------------------------------------------------------------------
// DatasetModel::DatasetParameter::getValue
//----------------------------------------------------------------------------
std::string DatasetModel::DatasetParameter::getValue(IDataset* dataset_p) const
{
  CDMA_FUNCTION_TRACE("DatasetModel::DatasetParameter::getValue");

  if( CONSTANT == m_type )
    return m_value;

  IContainerPtr ptr_container = dataset_p->findContainerByPath(m_target_path);

  switch( m_type )
  {
    case EXIST:
      if( ptr_container )
        return "true";
      return "false";

    case EQUAL:
      if( ptr_container->getContainerType() == IContainer::DATA_ITEM )
      {
        yat::String value = IDataItemPtr(ptr_container)->getValue<std::string>();
        if( value.is_equal(m_value) )
          return "true";
        return "false";
      }
      else
      {
        THROW_EXCEPTION( "BAD_CONFIG", 
                         PSZ_FMT("Bad path for dataset parameter"),
                         "DatasetModel::DatasetParameter::getValue" );
      }

    case VALUE:
      if( ptr_container->getContainerType() == IContainer::DATA_ITEM )
        return IDataItemPtr(ptr_container)->getValue<std::string>();
      else
      {
        THROW_EXCEPTION( "BAD_CONFIG", 
                         PSZ_FMT("Bad path for dataset parameter"),
                         "DatasetModel::DatasetParameter::getValue" );
      }

    case NAME:
      CDMA_TRACE("NAME");
      return ptr_container->getName();

    default:
        THROW_EXCEPTION( "BAD_CONFIG", 
                         PSZ_FMT("Bad dataset parameter type"),
                         "DatasetModel::DatasetParameter::getValue" );
  }
}

//----------------------------------------------------------------------------
// DatasetModel::Criterion::doTest
//----------------------------------------------------------------------------
bool DatasetModel::Criterion::doTest(IDataset* dataset_p) const
{
  CDMA_FUNCTION_TRACE("DatasetModel::Criterion::doTest");

  IContainerPtr ptr_container = dataset_p->findContainerByPath(m_target_path);

  switch( m_type )
  {
    case EXIST:
      {
        bool result = ptr_container ? true : false;
        CDMA_TRACE(m_target_path << " -> " << result << "(" << yat::any_cast<bool>(m_criterion_value) << ")");

        if( result && yat::any_cast<bool>(m_criterion_value) )
          return true;
        if( !result && !yat::any_cast<bool>(m_criterion_value) )
          return true;
        return false;
      }

    case EQUAL:
      if( ptr_container->getContainerType() == IContainer::DATA_ITEM )
      {
        yat::String value = IDataItemPtr(ptr_container)->getValue<std::string>();
        CDMA_TRACE("value: " << value);
        if( value.is_equal_no_case( yat::any_cast<std::string>(m_criterion_value) ) )
          return "true";
        return "false";
      }
      else
      {
        THROW_EXCEPTION( "BAD_CONFIG", 
                         PSZ_FMT("Bad path for dataset parameter"),
                         "DatasetModel::Criterion::doTest" );
      }

    case NOT_EQUAL:
      if( ptr_container->getContainerType() == IContainer::DATA_ITEM )
      {
        yat::String value = IDataItemPtr(ptr_container)->getValue<std::string>();
        if( !( value.is_equal( yat::any_cast<std::string>(m_criterion_value) ) ) )
          return "true";
        return "false";
      }
      else
      {
        THROW_EXCEPTION( "BAD_CONFIG", 
                         PSZ_FMT("Bad path for dataset parameter"),
                         "DatasetModel::Criterion::doTest" );
      }

    default:
        THROW_EXCEPTION( "BAD_CONFIG", 
                         PSZ_FMT("Bad criterion type"),
                         "DatasetModel::Criterion::doTest" );
  }
}

//----------------------------------------------------------------------------
// DatasetModel::applyCriteria
//----------------------------------------------------------------------------
bool DatasetModel::applyCriteria(IDataset* dataset_p) const
{
  CDMA_FUNCTION_TRACE("PluginConfigManager::applyCriteria");
  for( std::list<Criterion>::const_iterator cit = m_list_criteria.begin();
       cit != m_list_criteria.end(); ++cit )
  {
    if( !cit->doTest(dataset_p) )
      return false;
  }
  CDMA_TRACE("ok " << m_name);
  return true;
}

//----------------------------------------------------------------------------
// DatasetModel::getAllParameters
//----------------------------------------------------------------------------
void DatasetModel::getAllParameters(IDataset* dataset_p, DatasetModel::ParamMap *params_p) const
{
  CDMA_FUNCTION_TRACE("PluginConfigManager::getAllParameters");
  // Collect the plugins parameters
  *params_p = m_map_plugin_parameter;

  // Collect the dataset parameters
  for( std::map<std::string, DatasetParameter>::const_iterator cit = m_map_dataset_parameter.begin();
       cit != m_map_dataset_parameter.end(); ++cit )
  {
    (*params_p)[cit->first] = cit->second.getValue(dataset_p);
    CDMA_TRACE(cit->first << ": " << (*params_p)[cit->first]);
  }
}

//----------------------------------------------------------------------------
// PluginConfigManager::load
//----------------------------------------------------------------------------
void PluginConfigManager::load(const std::string& plugin_id,
                               const std::string& file_name) 
                                               throw ( cdma::Exception )
{
  CDMA_STATIC_FUNCTION_TRACE("PluginConfigManager::load");
  try
  {
    PluginConfigAnalyser analyser( &instance(), plugin_id );
    SAXParsor::start( Factory::getDictionariesFolder() + file_name, &analyser );
  }
  catch( yat::Exception& e )
  {
    e.push_error("READ_ERROR", "Cannot read plugin configuration file", 
                 "PluginConfigManager::loadConfigFile");
    LOG_EXCEPTION("cdma", e);
    RE_THROW_EXCEPTION(e);
  }
}

//----------------------------------------------------------------------------
// PluginConfigManager::getConfiguration
//----------------------------------------------------------------------------
void PluginConfigManager::getConfiguration(const std::string& plugin_id,
                                           IDataset* dataset_p, 
                                           DatasetModel::ParamMap* params_p ) throw ( cdma::Exception )
{
  CDMA_STATIC_FUNCTION_TRACE("PluginConfigManager::getConfiguration");
  DatasetModelList listModels = instance().m_map_model[plugin_id];
  for( DatasetModelList::const_iterator cit = listModels.begin(); cit != listModels.end(); ++cit )
  {
    if( (*cit)->applyCriteria(dataset_p) )
    {
      (*cit)->getAllParameters(dataset_p, params_p);
      break;
    }
  }
}

} //namespace



