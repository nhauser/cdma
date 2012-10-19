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
#include <yat/file/FileName.h>
#include <yat/utils/URI.h>

// CDMA core
#include <cdma/dictionary/IKey.h>
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/dictionary/impl/LogicalGroup.h>
#include <cdma/utils/PluginConfig.h>

// Soleil plugin
#include <SoleilNxsFactory.h>
#include <SoleilNxsDataSource.h>
#include <SoleilNxsDataset.h>

namespace cdma
{
namespace soleil
{
namespace nexus
{

//---------------------------------------------------------------------------
// Dataset::Dataset
//---------------------------------------------------------------------------
Dataset::Dataset( const yat::URI& location, Factory *factory_ptr ):
cdma::nexus::Dataset( yat::URI("file:" + location.get(yat::URI::PATH)), factory_ptr )
{
  // Utiliser yat pour sortir un FileName et récupérer son contenu (1 ou plusieurs fichiers)
  m_location = location;
}

//---------------------------------------------------------------------------
// Dataset::Dataset
//---------------------------------------------------------------------------
Dataset::Dataset()
: cdma::nexus::Dataset()
{
}

//---------------------------------------------------------------------------
// Dataset::getRootPath
//---------------------------------------------------------------------------
std::string Dataset::getRootPath() const
{
  return m_location.get(yat::URI::FRAGMENT);
}

//---------------------------------------------------------------------------
// Dataset::getItemFromPath
//---------------------------------------------------------------------------
IDataItemPtr Dataset::getItemFromPath(const std::string &fullPath)
{
  CDMA_FUNCTION_TRACE("cdma::soleil::nexus::Dataset::getItemFromPath(const std::string &)");
  return cdma::nexus::Dataset::getItemFromPath(getRootPath() + fullPath);
}

IDataItemPtr Dataset::getItemFromPath(const yat::String& path, const yat::String& name)
{
  CDMA_FUNCTION_TRACE("cdma::soleil::nexus::Dataset::getItemFromPath(const yat::String&, const yat::String&)");
  return cdma::nexus::Dataset::getItemFromPath(getRootPath() + path, name);
}

//---------------------------------------------------------------------------
// Dataset::findContainerByPath
//---------------------------------------------------------------------------
IContainerPtr Dataset::findContainerByPath(const std::string& input_path)
{
  CDMA_FUNCTION_TRACE("cdma::soleil::nexus::Dataset::findContainerByPath");
  IContainerPtrList containers = findAllContainerByPath( getRootPath() + input_path, true);
  if( containers.begin() != containers.end() )
    return *(containers.begin());
  return NULL;
}

//---------------------------------------------------------------------------
// Dataset::getLogicalRoot
//---------------------------------------------------------------------------
cdma::ILogicalGroupPtr Dataset::getLogicalRoot()
{
  FUNCTION_TRACE("Dataset::getLogicalRoot");

  if( m_log_root.is_null() )
  {
    CDMA_TRACE("Creating Dictionary detector");
    DictionaryDetector detector ( m_file_handle );
    CDMA_TRACE("Getting mapping file");

    // Load plugin configuration file
    PluginConfigManager::load( Factory::plugin_id(), "mappings/SoleilNeXus/cdma_nexussoleil_config.xml" );

    cdma::DatasetModel::ParamMap map_params;
    PluginConfigManager::getConfiguration(Factory::plugin_id(), this, &map_params);

    // Get mapping file parameters
    yat::String beamline = map_params["BEAMLINE"];
    yat::String model = map_params["MODEL"];

    if( beamline.empty() || model.empty() )
    {
      THROW_NO_RESULT("No mapping found for this dataset", "cdma::soleil::nexus::Dataset::getLogicalRoot");
    }

    // Mapping file name is lower case
    beamline.to_lower();
    model.to_lower();

    std::string mapping_file = beamline + "_" + model + ".xml";

    CDMA_TRACE("Creating dictionary " << mapping_file);
    cdma::DictionaryPtr dictionary ( new cdma::Dictionary(Factory::plugin_id()) );
    dictionary->setMappingFileName( mapping_file );

    CDMA_TRACE("Read the dictionary");
    dictionary->readEntries();

    CDMA_TRACE("Creating logical root");
    cdma::ILogicalGroup* ptrRoot = new cdma::LogicalGroup( this, NULL, IKeyPtr(NULL), dictionary );
    m_log_root.reset( ptrRoot );
  }
  return m_log_root;
}

} // namespace nexus
} // namespace soleil
} // namespace cdma
