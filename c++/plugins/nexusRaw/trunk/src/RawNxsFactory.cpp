// ****************************************************************************
// Synchrotron Raw
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
#include <yat/utils/URI.h>

// CDMA core
#include <cdma/Common.h>
#include <cdma/dictionary/Key.h>

// NeXus Engine
#include <NxsDataset.h>

// Raw plugin
#include <RawNxsFactory.h>

EXPORT_SINGLECLASS_PLUGIN(cdma::RawNxsFactory, \
                          cdma::RawNxsFactoryInfo);

namespace cdma
{

//----------------------------------------------------------------------------
// RawNxsFactoryInfo::get_plugin_id
//----------------------------------------------------------------------------
std::string RawNxsFactoryInfo::get_plugin_id() const
{
  return RawNxsFactory::plugin_id();
}

//----------------------------------------------------------------------------
// RawNxsFactoryInfo::get_interface_name
//----------------------------------------------------------------------------
std::string RawNxsFactoryInfo::get_interface_name() const
{
  return RawNxsFactory::interface_name();
}

//----------------------------------------------------------------------------
// RawNxsFactoryInfo::get_version_number
//----------------------------------------------------------------------------
std::string RawNxsFactoryInfo::get_version_number() const
{
  return RawNxsFactory::version_number();
}

//==============================================================================
// class RawNxsFactory
//==============================================================================
//----------------------------------------------------------------------------
// RawNxsFactory::RawNxsFactory
//----------------------------------------------------------------------------
RawNxsFactory::RawNxsFactory()
{
  CDMA_FUNCTION_TRACE("RawNxsFactory::RawNxsFactory");
}

//----------------------------------------------------------------------------
// RawNxsFactory::~RawNxsFactory
//----------------------------------------------------------------------------
RawNxsFactory::~RawNxsFactory()
{
  CDMA_TRACE("RawNxsFactory::~RawNxsFactory");
}

//----------------------------------------------------------------------------
// RawNxsFactory::openDataset
//----------------------------------------------------------------------------
IDatasetPtr RawNxsFactory::openDataset(const std::string& location)
throw ( cdma::Exception )
{
  return new RawNxsDataset( yat::URI(location), this );
}

//----------------------------------------------------------------------------
// RawNxsFactory::openDictionary
//----------------------------------------------------------------------------
DictionaryPtr RawNxsFactory::openDictionary(const std::string&)
throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("RawNxsFactory::openDictionary");
}

//----------------------------------------------------------------------------
// RawNxsFactory::createDatasetInstance
//----------------------------------------------------------------------------
IDatasetPtr RawNxsFactory::createDatasetInstance(const std::string&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("RawNxsFactory::createDatasetInstance");
}

//----------------------------------------------------------------------------
// RawNxsFactory::createEmptyDatasetInstance
//----------------------------------------------------------------------------
IDatasetPtr RawNxsFactory::createEmptyDatasetInstance() throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("RawNxsFactory::createEmptyDatasetInstance");
  THROW_NOT_IMPLEMENTED("RawNxsFactory::createEmptyDatasetInstance");
}

//----------------------------------------------------------------------------
// RawNxsFactory::getPathSeparator
//----------------------------------------------------------------------------
std::string RawNxsFactory::getPathSeparator()
{
  return std::string();
}

//----------------------------------------------------------------------------
// RawNxsFactory::getPluginURIDetector
//----------------------------------------------------------------------------
IDataSourcePtr RawNxsFactory::getPluginURIDetector()
{
  return IDataSourcePtr(NULL);
}

//----------------------------------------------------------------------------
// RawNxsFactory::getPluginMethodsList
//----------------------------------------------------------------------------
std::list<std::string> RawNxsFactory::getPluginMethodsList()
{
  return std::list<std::string>();
}

//==============================================================================
// class RawNxsDataset
//==============================================================================
//---------------------------------------------------------------------------
// RawNxsDataset::RawNxsDataset
//---------------------------------------------------------------------------
RawNxsDataset::RawNxsDataset( const yat::URI& location, 
                              RawNxsFactory* factory_ptr )
: NxsDataset( location, factory_ptr )
{
}

//---------------------------------------------------------------------------
// RawNxsDataset::RawNxsDataset
//---------------------------------------------------------------------------
RawNxsDataset::RawNxsDataset()
: NxsDataset()
{
}

//---------------------------------------------------------------------------
// RawNxsDataset::getLogicalRoot
//---------------------------------------------------------------------------
LogicalGroupPtr RawNxsDataset::getLogicalRoot()
{
  return LogicalGroupPtr(NULL);
}

} // namespace cdma
