// ****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : See AUTHORS file
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// ****************************************************************************

// Yat
#include <yat/plugin/PlugInSymbols.h>

// CDMA core
#include <cdma/Common.h>
#include <cdma/dictionary/Key.h>

// Engine NeXus
#include <NxsDataset.h>

// Soleil plugin
#include <SoleilNxsFactory.h>
#include <SoleilNxsDataSource.h>
#include <SoleilNxsDataset.h>

// Plugin declaration
EXPORT_SINGLECLASS_PLUGIN(cdma::soleil::nexus::Factory, \
                          cdma::soleil::nexus::FactoryInfo);

namespace cdma
{
namespace soleil
{
namespace nexus
{

//----------------------------------------------------------------------------
// FactoryInfo::get_plugin_id
//----------------------------------------------------------------------------
std::string FactoryInfo::get_plugin_id() const
{
  return Factory::plugin_id();
}

//----------------------------------------------------------------------------
// FactoryInfo::get_interface_name
//----------------------------------------------------------------------------
std::string FactoryInfo::get_interface_name() const
{
  FUNCTION_TRACE("Factory::get_interface_name");
  return Factory::interface_name();
}

//----------------------------------------------------------------------------
// FactoryInfo::get_version_number
//----------------------------------------------------------------------------
std::string FactoryInfo::get_version_number() const
{
  return Factory::version_number();
}
//==============================================================================
// class Factory
//==============================================================================
//----------------------------------------------------------------------------
// Factory::Factory
//----------------------------------------------------------------------------
Factory::Factory()
{
  FUNCTION_TRACE("Factory::Factory");
}

//----------------------------------------------------------------------------
// Factory::~Factory
//----------------------------------------------------------------------------
Factory::~Factory()
{
  FUNCTION_TRACE("Factory::~Factory");
}

//----------------------------------------------------------------------------
// Factory::openDataset
//----------------------------------------------------------------------------
cdma::IDatasetPtr Factory::openDataset(const std::string& location_string)
throw ( cdma::Exception )
{
  FUNCTION_TRACE("Factory::openDataset");
  return new Dataset( yat::URI(location_string), this );
}

//----------------------------------------------------------------------------
// Factory::openDictionary
//----------------------------------------------------------------------------
DictionaryPtr Factory::openDictionary(const std::string&)
throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::openDictionary");
}

//----------------------------------------------------------------------------
// Factory::createDatasetInstance
//----------------------------------------------------------------------------
cdma::IDatasetPtr Factory::createDatasetInstance(const std::string&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::createDatasetInstance");
}

//----------------------------------------------------------------------------
// Factory::createEmptyDatasetInstance
//----------------------------------------------------------------------------
cdma::IDatasetPtr Factory::createEmptyDatasetInstance() throw ( cdma::Exception )
{
  FUNCTION_TRACE("Factory::createEmptyDatasetInstance");
  THROW_NOT_IMPLEMENTED("Factory::createEmptyDatasetInstance");
}

//----------------------------------------------------------------------------
// Factory::getPathSeparator
//----------------------------------------------------------------------------
std::string Factory::getPathSeparator()
{
  THROW_NOT_IMPLEMENTED("Factory::getPathSeparator");
}

//----------------------------------------------------------------------------
// Factory::getPluginURIDetector
//----------------------------------------------------------------------------
IDataSourcePtr Factory::getPluginURIDetector()
{
  return new DataSource(this);
}

//----------------------------------------------------------------------------
// Factory::getPluginMethodsList
//----------------------------------------------------------------------------
std::list<std::string> Factory::getPluginMethodsList()
{
  std::list<std::string> methods;
  methods.push_back("TestMethod");
  return methods;
}

} // namespace nexus
} // namespace soleil
} // namespace cdma
