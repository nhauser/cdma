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
// yat
#include <yat/utils/String.h>
#include <yat/utils/Logging.h>
#include <yat/utils/URI.h>
#include <yat/file/FileName.h>

// cdma
#include <cdma/exception/impl/ExceptionImpl.h>
#include <cdma/dictionary/impl/Key.h>
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/dictionary/plugin/Context.h>
#include <cdma/IDataSource.h>
#include <cdma/factory/plugin/IPluginFactory.h>
#include <cdma/factory/Factory.h>
#include <cdma/factory/impl/FactoryImpl.h>

namespace cdma
{

//----------------------------------------------------------------------------
// Factory::cleanup
//----------------------------------------------------------------------------
void Factory::cleanup()
{
  FactoryImpl::cleanup();
}

//----------------------------------------------------------------------------
// Factory::init
//----------------------------------------------------------------------------
void Factory::init(const std::string &plugin_path)
{
  FactoryImpl::init(plugin_path);
}

//----------------------------------------------------------------------------
// Factory::setActiveView
//----------------------------------------------------------------------------
void Factory::setActiveView(const std::string& experiment)
{
  FactoryImpl::setActiveView(experiment);
}

//----------------------------------------------------------------------------
// Factory::getActiveView
//----------------------------------------------------------------------------
const std::string& Factory::getActiveView()
{
  return FactoryImpl::getActiveView();
}

//----------------------------------------------------------------------------
// Factory::getKeyDictionaryPath
//----------------------------------------------------------------------------
std::string Factory::getKeyDictionaryPath()
{
  return FactoryImpl::getKeyDictionaryPath();
}

//----------------------------------------------------------------------------
// Factory::getConceptDictionaryFolder
//----------------------------------------------------------------------------
std::string Factory::getConceptDictionaryFolder()
{
  return FactoryImpl::getConceptDictionaryFolder();
}

//----------------------------------------------------------------------------
// Factory::getKeyDictionaryFolder
//----------------------------------------------------------------------------
std::string Factory::getKeyDictionaryFolder()
{
  return FactoryImpl::getKeyDictionaryFolder();
}

//----------------------------------------------------------------------------
// Factory::getMappingDictionaryFolder
//----------------------------------------------------------------------------
std::string Factory::getMappingDictionaryFolder(const std::string& plugin_id)
{
  return FactoryImpl::getMappingDictionaryFolder(plugin_id);
}
    
//----------------------------------------------------------------------------
// Factory::getDictionariesFolder
//----------------------------------------------------------------------------
std::string Factory::getDictionariesFolder()
{
  return FactoryImpl::getDictionariesFolder();
}

//----------------------------------------------------------------------------
// Factory::openDataset
//----------------------------------------------------------------------------
IDatasetPtr Factory::openDataset( const std::string& uri ) throw ( Exception )
{
  try
  {
    return FactoryImpl::openDataset(yat::URI(uri));
  }
  catch( yat::Exception& e )
  {
    RE_THROW_EXCEPTION(e);
  }
}

//----------------------------------------------------------------------------
// Factory::openDictionary
//----------------------------------------------------------------------------
DictionaryPtr Factory::openDictionary( const std::string& ) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::openDictionary");
}

//----------------------------------------------------------------------------
// Factory::getDataSource
//----------------------------------------------------------------------------
IDataSourcePtr Factory::getDataSource(const std::string& uri) throw ( Exception )
{
  try
  {
    return FactoryImpl::getDataSource( yat::URI(uri) );
  }
  catch( yat::Exception& e )
  {
    RE_THROW_EXCEPTION(e);
  }
}

//----------------------------------------------------------------------------
// Factory::createView
//----------------------------------------------------------------------------
IViewPtr Factory::createView(std::vector<int> shape, std::vector<int> start, std::vector<int> stride)
{
  return new View( shape, start, stride );
}

//----------------------------------------------------------------------------
// Factory::createView
//----------------------------------------------------------------------------
IKeyPtr Factory::createKey(const std::string& name, IKey::Type type)
{
  return new Key(name, type);
}

} // namespace cdma
