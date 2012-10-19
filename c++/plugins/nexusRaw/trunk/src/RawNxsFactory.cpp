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
#include <yat/file/FileName.h>

// CDMA core
#include <cdma/Common.h>
#include <cdma/dictionary/IKey.h>

// NeXus Engine
#include <NxsDataset.h>

// Raw plugin
#include <RawNxsFactory.h>

EXPORT_SINGLECLASS_PLUGIN(cdma::soleil::rawnexus::Factory, \
                          cdma::soleil::rawnexus::FactoryInfo);

namespace cdma
{

namespace soleil
{

namespace rawnexus
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
  CDMA_FUNCTION_TRACE("cdma::soleil::rawnexus::Factory::Factory");
}

//----------------------------------------------------------------------------
// Factory::~Factory
//----------------------------------------------------------------------------
Factory::~Factory()
{
  CDMA_TRACE("cdma::soleil::rawnexus::Factory::~Factory");
}

//----------------------------------------------------------------------------
// Factory::openDataset
//----------------------------------------------------------------------------
cdma::IDatasetPtr Factory::openDataset(const std::string& location)
throw ( cdma::Exception )
{
  return new Dataset( yat::URI(location), this );
}

//----------------------------------------------------------------------------
// Factory::openDictionary
//----------------------------------------------------------------------------
cdma::DictionaryPtr Factory::openDictionary(const std::string&)
throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("cdma::soleil::rawnexus::Factory::openDictionary");
}

//----------------------------------------------------------------------------
// Factory::createDatasetInstance
//----------------------------------------------------------------------------
cdma::IDatasetPtr Factory::createDatasetInstance(const std::string&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("cdma::soleil::rawnexus::Factory::createDatasetInstance");
}

//----------------------------------------------------------------------------
// Factory::createEmptyDatasetInstance
//----------------------------------------------------------------------------
cdma::IDatasetPtr Factory::createEmptyDatasetInstance() throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("cdma::soleil::rawnexus::Factory::createEmptyDatasetInstance");
  THROW_NOT_IMPLEMENTED("cdma::soleil::rawnexus::Factory::createEmptyDatasetInstance");
}

//----------------------------------------------------------------------------
// Factory::getPathSeparator
//----------------------------------------------------------------------------
std::string Factory::getPathSeparator()
{
  return std::string();
}

//----------------------------------------------------------------------------
// Factory::getPluginURIDetector
//----------------------------------------------------------------------------
cdma::IDataSourcePtr Factory::getPluginURIDetector()
{
  return new DataSource(this);
}

//----------------------------------------------------------------------------
// Factory::getPluginMethodsList
//----------------------------------------------------------------------------
std::list<std::string> Factory::getPluginMethodsList()
{
  return std::list<std::string>();
}

//==============================================================================
// class Dataset
//==============================================================================
//---------------------------------------------------------------------------
// Dataset::Dataset
//---------------------------------------------------------------------------
Dataset::Dataset( const yat::URI& location, Factory* factory_ptr )
: cdma::nexus::Dataset( location, factory_ptr )
{
}

//---------------------------------------------------------------------------
// Dataset::Dataset
//---------------------------------------------------------------------------
Dataset::Dataset()
: cdma::nexus::Dataset()
{
}

//---------------------------------------------------------------------------
// Dataset::getLogicalRoot
//---------------------------------------------------------------------------
cdma::ILogicalGroupPtr Dataset::getLogicalRoot()
{
  return cdma::ILogicalGroupPtr(NULL);
}

//==============================================================================
// class DataSource
//==============================================================================
//----------------------------------------------------------------------------
// DataSource::isReadable
//----------------------------------------------------------------------------
bool DataSource::isReadable(const std::string& dataset_uri) const
{
  CDMA_FUNCTION_TRACE("cdma::soleil::rawnexus::DataSource::isReadable");
  // Get the path from URI
  yat::URI dataset_location(dataset_uri);
  yat::String path = dataset_location.get( yat::URI::PATH );
  
  // Check file exists and is has a NeXus extension
  yat::FileName file ( path );
  
  if( file.file_exist() )
  {
    try
    {
      // Will try to open the file and close it
      CDMA_TRACE("try to open dataset " << dataset_location.get());
      m_factory_ptr->openDataset( dataset_location.get() );
      CDMA_TRACE("return true");
      return true;
    }
    catch( ... )
    {
      CDMA_TRACE("return false");
      return false;
    }
  }
  CDMA_TRACE("return false");
  return false; 
}

//----------------------------------------------------------------------------
// DataSource::isBrowsable
//----------------------------------------------------------------------------
bool DataSource::isBrowsable( const std::string& ) const
{
  return false;
}

//----------------------------------------------------------------------------
// DataSource::isProducer
//----------------------------------------------------------------------------
bool DataSource::isProducer( const std::string& ) const
{
  return false;
}

//----------------------------------------------------------------------------
// DataSource::isExperiment
//----------------------------------------------------------------------------
bool DataSource::isExperiment( const std::string& ) const
{
  return false;
}

} // namespace rawnexus
} // namespace soleil
} // namespace cdma
