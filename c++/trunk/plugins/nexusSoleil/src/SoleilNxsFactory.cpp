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

EXPORT_SINGLECLASS_PLUGIN(cdma::SoleilNxsFactory, \
                          cdma::SoleilNxsFactoryInfo);

namespace cdma
{

const std::string PlugInID         ( "SoleilNeXus" );
const std::string InterfaceName    ( "cdma::IFactory" );
const std::string VersionNumber    ( "1.0.0" );

//----------------------------------------------------------------------------
// SoleilNxsFactoryInfo::get_plugin_id
//----------------------------------------------------------------------------
std::string SoleilNxsFactoryInfo::get_plugin_id() const
{
  return PlugInID;
}

//----------------------------------------------------------------------------
// SoleilNxsFactoryInfo::get_interface_name
//----------------------------------------------------------------------------
std::string SoleilNxsFactoryInfo::get_interface_name() const
{
  return InterfaceName;
}

//----------------------------------------------------------------------------
// SoleilNxsFactoryInfo::get_version_number
//----------------------------------------------------------------------------
std::string SoleilNxsFactoryInfo::get_version_number() const
{
  return VersionNumber;
}

//==============================================================================
// class SoleilNxsFactory
//==============================================================================
//----------------------------------------------------------------------------
// SoleilNxsFactory::SoleilNxsFactory
//----------------------------------------------------------------------------
SoleilNxsFactory::SoleilNxsFactory()
{
  CDMA_FUNCTION_TRACE("SoleilNxsFactory::SoleilNxsFactory");
}

//----------------------------------------------------------------------------
// SoleilNxsFactory::~SoleilNxsFactory
//----------------------------------------------------------------------------
SoleilNxsFactory::~SoleilNxsFactory()
{
  CDMA_FUNCTION_TRACE("SoleilNxsFactory::~SoleilNxsFactory");
}

//----------------------------------------------------------------------------
// SoleilNxsFactory::openDataset
//----------------------------------------------------------------------------
IDatasetPtr SoleilNxsFactory::openDataset(const std::string& location_string)
throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("SoleilNxsFactory::openDataset");
  return SoleilNxsDataset::getDataset( yat::URI(location_string), this);
}

//----------------------------------------------------------------------------
// SoleilNxsFactory::openDictionary
//----------------------------------------------------------------------------
DictionaryPtr SoleilNxsFactory::openDictionary(const std::string&)
throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("SoleilNxsFactory::openDictionary");
}

//----------------------------------------------------------------------------
// SoleilNxsFactory::createDatasetInstance
//----------------------------------------------------------------------------
IDatasetPtr SoleilNxsFactory::createDatasetInstance(const std::string&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("SoleilNxsFactory::createDatasetInstance");
}

//----------------------------------------------------------------------------
// SoleilNxsFactory::createEmptyDatasetInstance
//----------------------------------------------------------------------------
IDatasetPtr SoleilNxsFactory::createEmptyDatasetInstance() throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("SoleilNxsFactory::createEmptyDatasetInstance");
  THROW_NOT_IMPLEMENTED("SoleilNxsFactory::createEmptyDatasetInstance");
}

//----------------------------------------------------------------------------
// SoleilNxsFactory::getPathSeparator
//----------------------------------------------------------------------------
std::string SoleilNxsFactory::getPathSeparator()
{
  THROW_NOT_IMPLEMENTED("SoleilNxsFactory::getPathSeparator");
}

//----------------------------------------------------------------------------
// SoleilNxsFactory::getPluginURIDetector
//----------------------------------------------------------------------------
IDataSourcePtr SoleilNxsFactory::getPluginURIDetector()
{
  return new SoleilNxsDataSource();
}

//----------------------------------------------------------------------------
// SoleilNxsFactory::getPluginMethodsList
//----------------------------------------------------------------------------
std::list<std::string> SoleilNxsFactory::getPluginMethodsList()
{
  std::list<std::string> methods;
  methods.push_back("TestMethod");
  return methods;
}

} // namespace cdma
