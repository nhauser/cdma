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
#ifdef YAT_WIN32
  const std::string SHARED_LIB_EXTENSION( "dll" );
#else
  const std::string SHARED_LIB_EXTENSION( "so" );
#endif

std::string INTERFACE_NAME("cdma::IPluginFactory");
std::string FactoryImpl::s_dico_path_prop = "CDMA_DICTIONARY_PATH";
std::string FactoryImpl::s_cdma_view = "";

//----------------------------------------------------------------------------
// FactoryImpl::instance
//----------------------------------------------------------------------------
FactoryImpl& FactoryImpl::instance()
{
  static FactoryImpl the_instance;
  return the_instance;
}

//----------------------------------------------------------------------------
// FactoryImpl::cleanup
//----------------------------------------------------------------------------
void FactoryImpl::cleanup()
{
  CDMA_STATIC_FUNCTION_TRACE("FactoryImpl::cleanup");
  // Free method objects for each plugins
  for( PluginMap::iterator it = instance().m_plugin_map.begin();
                           it != instance().m_plugin_map.end(); it++)
  {
    it->second.plugin_method_map.clear();
  }

  CDMA_STATIC_TRACE("clear factory map");

  // Release plugins
  instance().m_plugin_factory_map.clear();
  instance().m_plugin_map.clear();
}

//----------------------------------------------------------------------------
// FactoryImpl::init
//----------------------------------------------------------------------------
void FactoryImpl::init(const std::string &plugin_path)
{
  CDMA_STATIC_FUNCTION_TRACE("FactoryImpl::init");

  yat::FileEnum fe(plugin_path + '/', yat::FileEnum::ENUM_FILE);
  CDMA_STATIC_TRACE(fe.path());

  if( !fe.path_exist() )
    THROW_EXCEPTION("BAD_PATH", PSZ_FMT("Path not exists (%s)", PSZ(plugin_path)), "FactoryImpl::init");
  
  while( fe.find() )
  {
    CDMA_STATIC_TRACE(fe.name_ext() << " " << SHARED_LIB_EXTENSION);

    if( fe.ext().is_equal(SHARED_LIB_EXTENSION) )
    {
      //- we found a shared lib
      Plugin plugin_data;

      CDMA_STATIC_TRACE("Found lib " << fe.full_name());
      try
      {
        // Load the plugin
        yat::PlugInManager::PlugInEntry plugin_entry;
        instance().m_plugin_manager.load( fe.full_name(), &plugin_entry );
        plugin_data.info = plugin_entry.m_info;
        plugin_data.factory = plugin_entry.m_factory;
        plugin_data.plugin_objet = plugin_entry.m_plugin;
        CDMA_STATIC_TRACE("Plug-in loaded");
      }
      catch( yat::Exception& e)
      {
        e.dump();
        continue;
      }
      CDMA_STATIC_TRACE("plugin_data.info->get_interface_name: " << plugin_data.info->get_interface_name());
      
      yat::IPlugInInfo* plugin_info = plugin_data.info;
      if( plugin_info->get_interface_name() == INTERFACE_NAME )
      {
        // We found a CDMA plugin!
        CDMA_STATIC_TRACE("Found a CDMA plugin!");
        
        std::string plugin_id = plugin_info->get_plugin_id();
        CDMA_STATIC_TRACE("Plugin_objects.info->get_plugin_id: " << plugin_id);
        
        instance().m_plugin_map[plugin_id] = plugin_data;
        CDMA_STATIC_TRACE("Plugin registered");
      }
    }
  }
}

//----------------------------------------------------------------------------
// FactoryImpl::initPluginMethods
//----------------------------------------------------------------------------
void FactoryImpl::initPluginMethods(const IPluginFactoryPtr& factory_ptr, FactoryImpl::Plugin *plugin_ptr)
{
  CDMA_FUNCTION_TRACE("FactoryImpl::initPluginMethods");
  CDMA_TRACE("Plugin id: " << plugin_ptr->info->get_plugin_id());

  // Retreive the list of supported dictionary methods
  std::list<std::string> methods = factory_ptr->getPluginMethodsList();
  for( std::list<std::string>::iterator it = methods.begin(); it != methods.end(); it++ )
  {
    CDMA_TRACE("Method: " << *it);
    yat::String get_method_class = yat::String::str_format("get%sClass", PSZ(*it));
    try
    {
      yat::PlugIn::Symbol symbol = plugin_ptr->plugin_objet->find_symbol(get_method_class);
      GetMethodObject_t get_method_object_func = (GetMethodObject_t)(symbol);

      cdma::IPluginMethod* method_ptr = get_method_object_func();

      // Store the method object in association with the method name
#ifdef CDMA_STD_SMART_PTR
      plugin_ptr->plugin_method_map[*it] = cdma::IPluginMethodPtr(method_ptr);
#else
      plugin_ptr->plugin_method_map[*it] = method_ptr;
#endif

      //Context context;
      //method_ptr->execute(context);
    }
    catch( yat::Exception& ex )
    {
      RETHROW_YAT_ERROR(ex,
                      "METHOD_NOT_FOUND",
                      PSZ_FMT("Unable to find the '%s' symbol while initialize the '%s' plugin",
                              PSZ(get_method_class), PSZ(plugin_ptr->info->get_plugin_id())),
                      "PlugIn::factory");
    }
  }
}

//----------------------------------------------------------------------------
// FactoryImpl::getPluginFactory
//----------------------------------------------------------------------------
IPluginMethodPtr FactoryImpl::getPluginMethod(const std::string &plugin_id, 
                                          const std::string &method_name)
{
  // Find the plugin
  PluginMap::iterator plugin_it = instance().m_plugin_map.find(plugin_id);
  if( plugin_it == instance().m_plugin_map.end() )
    THROW_EXCEPTION( "NOT_FOUND", PSZ_FMT("No such plugin (%s).", PSZ(plugin_id)),
                           "cdma::FactoryImpl::getPluginMethod" );
  
  // Look for the requested method
  PluginMethodsMap& methods_map = instance().m_plugin_map[plugin_id].plugin_method_map;
  PluginMethodsMap::iterator method_it = methods_map.find(method_name);

  if( method_it == methods_map.end() )
    // Not found, returns a null pointer
    return IPluginMethodPtr(NULL);

  // ok
  return (*method_it).second;
}

//----------------------------------------------------------------------------
// FactoryImpl::getPluginFactory
//----------------------------------------------------------------------------
IPluginFactoryPtr FactoryImpl::getPluginFactory(const std::string &plugin_id)
{
  PluginFactoryPtrMap::iterator it = instance().m_plugin_factory_map.find(plugin_id);
  if( it == instance().m_plugin_factory_map.end() )
  {  
    // Find the plugin
    PluginMap::iterator it = instance().m_plugin_map.find(plugin_id);
    if( it == instance().m_plugin_map.end() )
      THROW_EXCEPTION( "NOT_FOUND", PSZ_FMT("No such plugin (%s).", PSZ(plugin_id)),
                             "cdma::FactoryImpl::getPluginFactory" );
    
    yat::IPlugInFactory* factory = (*it).second.factory;

    // Instanciate the IPluginFactory implementation
    yat::IPlugInObject* obj;
    factory->create(obj);
    IPluginFactoryPtr factory_ptr(dynamic_cast<cdma::IPluginFactory*>(obj));
    instance().m_plugin_factory_map[plugin_id] = factory_ptr;
    
    // Initialize the dictionnary methods supported by this plugin factory
    instance().initPluginMethods(factory_ptr, &(it->second));

    return factory_ptr;
  }
  
  // Factory already instancied
  return (*it).second;
}

//----------------------------------------------------------------------------
// FactoryImpl::setActiveView
//----------------------------------------------------------------------------
void FactoryImpl::setActiveView(const std::string& experiment)
{
  s_cdma_view = experiment;
}

//----------------------------------------------------------------------------
// FactoryImpl::getActiveView
//----------------------------------------------------------------------------
const std::string& FactoryImpl::getActiveView()
{
  return s_cdma_view;
}

//----------------------------------------------------------------------------
// FactoryImpl::getKeyDictionaryPath
//----------------------------------------------------------------------------
std::string FactoryImpl::getKeyDictionaryPath()
{
  yat::FileName file( getDictionariesFolder() + "/views/", ( getActiveView() + "_view.xml" ) );
  return file.full_name();
}

//----------------------------------------------------------------------------
// FactoryImpl::getConceptDictionaryFolder
//----------------------------------------------------------------------------
std::string FactoryImpl::getConceptDictionaryFolder()
{
  yat::FileName file( getDictionariesFolder() + "/concepts/" );
  return file.full_name();
}

//----------------------------------------------------------------------------
// FactoryImpl::getKeyDictionaryFolder
//----------------------------------------------------------------------------
std::string FactoryImpl::getKeyDictionaryFolder()
{
  yat::FileName file( getDictionariesFolder() + "/views/" );
  return file.full_name();
}

//----------------------------------------------------------------------------
// FactoryImpl::getMappingDictionaryFolder
//----------------------------------------------------------------------------
std::string FactoryImpl::getMappingDictionaryFolder(const std::string& plugin_id)
{
  yat::FileName file( getDictionariesFolder() + "/mappings/" + plugin_id + "/" );
  return file.full_name();
}
    
//----------------------------------------------------------------------------
// FactoryImpl::getDictionariesFolder
//----------------------------------------------------------------------------
std::string FactoryImpl::getDictionariesFolder()
{
  yat::String value;
  if( ! yat::SysUtils::get_env(s_dico_path_prop, &value, "") )
  {
    THROW_NO_RESULT("No environment variable '" + s_dico_path_prop + "' that defines dictionaries folder!", "FactoryImpl::getDictionariesFolder");
  }
  yat::FileName folder(value);
  return folder.full_name();
}

//----------------------------------------------------------------------------
// FactoryImpl::openDataset
//----------------------------------------------------------------------------
IDatasetPtr FactoryImpl::openDataset( const yat::URI& uri ) throw ( Exception )
{
  CDMA_STATIC_FUNCTION_TRACE("FactoryImpl::openDataset");
  std::pair< IDatasetPtr, IPluginFactoryPtr > result;
  
  try
  {
    IPluginFactoryPtr plugin = detectPluginFactory(uri);
    return plugin->openDataset( uri.get() );
  }
  catch( BaseException& e )
  {
    LOG_EXCEPTION("cdma", e);
    RE_THROW_EXCEPTION(e);
  }
  catch( yat::Exception& e )
  {
    LOG_EXCEPTION("yat", e);
    RE_THROW_EXCEPTION(e);
  }
}

//----------------------------------------------------------------------------
// FactoryImpl::openDictionary
//----------------------------------------------------------------------------
DictionaryPtr FactoryImpl::openDictionary( const std::string& ) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("FactoryImpl::openDictionary");
}

//----------------------------------------------------------------------------
// FactoryImpl::detectPluginFactory
//----------------------------------------------------------------------------
IPluginFactoryPtr FactoryImpl::detectPluginFactory(const yat::URI& uri) throw ( Exception )
{
  CDMA_STATIC_FUNCTION_TRACE("FactoryImpl::detectPluginFactory");
  IPluginFactoryPtr result;
  IPluginFactoryPtr tmp;
  IDataSourcePtr data_source;
  std::string plugin_id;
    
  // for each plugin that where found
  PluginMap::iterator iterator = instance().m_plugin_map.begin();
  while( iterator != instance().m_plugin_map.end() )
  {
    // Get the corresponding factory
    plugin_id = iterator->first;
    tmp = instance().getPluginFactory( plugin_id );
    data_source = tmp->getPluginURIDetector();
    
    if( !data_source )
    {
      yat::log_warning("cdma", "Plugin %s should implements IDataSource interface. Skip it.", PSZ(plugin_id));
    }
    else
    {
      // Ask if the URI is readable
      if( data_source->isReadable( uri.get() ) )
      {
        // Ask if the plugin is the owner of that URI
        if( data_source->isProducer( uri.get() ) )
        {
          result = tmp;
          break;
        }
        else
        {
          // Not owner but can read so keep it for later if no owner are found
          result = tmp;
        }
      }
    }
    ++iterator;
  }
  return result;
}

//----------------------------------------------------------------------------
// FactoryImpl::getDataSource
//----------------------------------------------------------------------------
IDataSourcePtr FactoryImpl::getDataSource(const yat::URI& uri) throw ( Exception )
{
  IPluginFactoryPtr plugin_factory = FactoryImpl::detectPluginFactory(uri);
  return plugin_factory->getPluginURIDetector();
}

} // namespace cdma
