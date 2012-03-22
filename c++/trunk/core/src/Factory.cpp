//*****************************************************************************
/// Synchrotron SOLEIL
///
/// Recording configuration parsor
///
/// Creation : 15/04/2005
/// Author   : S. Poirier
///
//*****************************************************************************

// yat
#include <yat/utils/String.h>
#include <yat/utils/Logging.h>
#include <yat/file/FileName.h>

// cdma
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/Dictionary.h>
#include <cdma/dictionary/Context.h>
#include <cdma/IDataSource.h>
#include <cdma/IFactory.h>
#include <cdma/Factory.h>

namespace cdma
{
#ifdef YAT_WIN32
  const std::string SHARED_LIB_EXTENSION( "dll" );
#else
  const std::string SHARED_LIB_EXTENSION( "so" );
#endif

std::string INTERFACE_NAME("cdma::IFactory");
std::string Factory::s_dico_path_prop = "CDMA_DICTIONARY_PATH";
std::string Factory::s_cdma_view = "";

//----------------------------------------------------------------------------
// Factory::instance
//----------------------------------------------------------------------------
Factory& Factory::instance()
{
  static Factory the_instance;
  return the_instance;
}

//----------------------------------------------------------------------------
// Factory::cleanup
//----------------------------------------------------------------------------
void Factory::cleanup()
{
  CDMA_STATIC_FUNCTION_TRACE("Factory::cleanup");
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
// Factory::init
//----------------------------------------------------------------------------
void Factory::init(const std::string &plugin_path)
{
  CDMA_STATIC_FUNCTION_TRACE("Factory::init");

  yat::FileEnum fe(plugin_path + '/', yat::FileEnum::ENUM_FILE);
  CDMA_STATIC_TRACE(fe.path());

  if( !fe.path_exist() )
    throw cdma::Exception("BAD_PATH", PSZ_FMT("Path not exists (%s)", PSZ(plugin_path)), "Factory::init");
  
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
// Factory::initPluginMethods
//----------------------------------------------------------------------------
void Factory::initPluginMethods(const IFactoryPtr& factory_ptr, Factory::Plugin *plugin_ptr)
{
  CDMA_FUNCTION_TRACE("Factory::initPluginMethods");
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
      plugin_ptr->plugin_method_map[*it] = method_ptr;

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
// Factory::getPluginFactory
//----------------------------------------------------------------------------
IPluginMethodPtr Factory::getPluginMethod(const std::string &plugin_id, 
                                          const std::string &method_name)
{
  // Find the plugin
  PluginMap::iterator plugin_it = instance().m_plugin_map.find(plugin_id);
  if( plugin_it == instance().m_plugin_map.end() )
    throw cdma::Exception( "NOT_FOUND", PSZ_FMT("No such plugin (%s).", PSZ(plugin_id)),
                           "cdma::Factory::getPluginMethod" );
  
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
// Factory::getPluginFactory
//----------------------------------------------------------------------------
IFactoryPtr Factory::getPluginFactory(const std::string &plugin_id)
{
  PluginFactoryPtrMap::iterator it = instance().m_plugin_factory_map.find(plugin_id);
  if( it == instance().m_plugin_factory_map.end() )
  {  
    // Find the plugin
    PluginMap::iterator it = instance().m_plugin_map.find(plugin_id);
    if( it == instance().m_plugin_map.end() )
      throw cdma::Exception( "NOT_FOUND", PSZ_FMT("No such plugin (%s).", PSZ(plugin_id)),
                             "cdma::Factory::getPluginFactory" );
    
    yat::IPlugInFactory* factory = (*it).second.factory;

    // Instanciate the IFactory implementation
    yat::IPlugInObject* obj;
    factory->create(obj);
    IFactoryPtr factory_ptr(dynamic_cast<cdma::IFactory*>(obj));
    instance().m_plugin_factory_map[plugin_id] = factory_ptr;
    
    // Initialize the dictionnary methods supported by this plugin factory
    instance().initPluginMethods(factory_ptr, &(it->second));

    return factory_ptr;
  }
  
  // Factory already instancied
  return (*it).second;
}

//----------------------------------------------------------------------------
// Factory::setActiveView
//----------------------------------------------------------------------------
void Factory::setActiveView(const std::string& experiment)
{
  s_cdma_view = experiment;
}

//----------------------------------------------------------------------------
// Factory::getActiveView
//----------------------------------------------------------------------------
const std::string& Factory::getActiveView()
{
  return s_cdma_view;
}

//----------------------------------------------------------------------------
// Factory::getKeyDictionaryPath
//----------------------------------------------------------------------------
std::string Factory::getKeyDictionaryPath()
{
  yat::FileName file( getDictionariesFolder(), ( getActiveView() + "_view.xml" ) );
  return file.full_name();
}

//----------------------------------------------------------------------------
// Factory::getMappingDictionaryFolder
//----------------------------------------------------------------------------
std::string Factory::getMappingDictionaryFolder(const IFactoryPtr& factory)
{
  yat::FileName file( getDictionariesFolder() + "/" + factory->getName() + "/" );
  return file.full_name();
}
    
//----------------------------------------------------------------------------
// Factory::getDictionariesFolder
//----------------------------------------------------------------------------
std::string Factory::getDictionariesFolder()
{
  yat::String value;
  if( ! yat::SysUtils::get_env(s_dico_path_prop, &value, "") )
  {
    THROW_NO_RESULT("No environment variable '" + s_dico_path_prop + "' that defines dictionaries folder!", "Factory::getDictionariesFolder");
  }
  yat::FileName folder(value);
  return folder.full_name();
}

//----------------------------------------------------------------------------
// Factory::openDataset
//----------------------------------------------------------------------------
std::pair<IDatasetPtr, IFactoryPtr> Factory::openDataset( const yat::URI& uri ) throw ( Exception )
{
  std::pair< IDatasetPtr, IFactoryPtr > result;
  
  result.second = detectPluginFactory(uri);
  
  result.first = result.second->openDataset( uri.get() );
  
  return result;
}

//----------------------------------------------------------------------------
// Factory::openDictionary
//----------------------------------------------------------------------------
DictionaryPtr Factory::openDictionary( const std::string& ) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::openDictionary");
}

//----------------------------------------------------------------------------
// Factory::createArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createArray( const std::type_info, const std::vector<int> )
{
  THROW_NOT_IMPLEMENTED("Factory::createArray");
}

//----------------------------------------------------------------------------
// Factory::createArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createArray( const void* )
{
  THROW_NOT_IMPLEMENTED("Factory::createArray");
}

//----------------------------------------------------------------------------
// Factory::createStringArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createStringArray( const std::string& )
{
  THROW_NOT_IMPLEMENTED("Factory::createStringArray");
}

//----------------------------------------------------------------------------
// Factory::createDoubleArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createDoubleArray( double [] )
{
  THROW_NOT_IMPLEMENTED("Factory::createDoubleArray");
}

//----------------------------------------------------------------------------
// Factory::createDoubleArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createDoubleArray( double [], const std::vector<int> )
{
  THROW_NOT_IMPLEMENTED("Factory::createDoubleArray");
}
/*
//----------------------------------------------------------------------------
// Factory::createDataItem
//----------------------------------------------------------------------------
IDataItemPtr Factory::createDataItem(const IGroupPtr& parent, const std::string& shortName, const ArrayPtr& array) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::createDataItem");
}

//----------------------------------------------------------------------------
// Factory::createGroup
//----------------------------------------------------------------------------
IGroupPtr Factory::createGroup(const IGroupPtr& parent, const std::string& shortName, const bool updateParent)
{
  THROW_NOT_IMPLEMENTED("Factory::createGroup");
}

//----------------------------------------------------------------------------
// Factory::createGroup
//----------------------------------------------------------------------------
IGroupPtr Factory::createGroup(const std::string& shortName) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::createGroup");
}

//----------------------------------------------------------------------------
// Factory::createAttribute
//----------------------------------------------------------------------------
IAttributePtr Factory::createAttribute(const std::string& name, const void * value)
{
  THROW_NOT_IMPLEMENTED("Factory::createAttribute");
}

//----------------------------------------------------------------------------
// Factory::createDatasetInstance
//----------------------------------------------------------------------------
IDatasetPtr Factory::createDatasetInstance(const std::string& uri) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::createDatasetInstance");
}

//----------------------------------------------------------------------------
// Factory::createEmptyDatasetInstance
//----------------------------------------------------------------------------
IDatasetPtr Factory::createEmptyDatasetInstance() throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::createEmptyDatasetInstance");
}
*/
//----------------------------------------------------------------------------
// Factory::createKey
//----------------------------------------------------------------------------
KeyPtr Factory::createKey( std::string )
{
  THROW_NOT_IMPLEMENTED("Factory::createKey");
}

//----------------------------------------------------------------------------
// Factory::detectPluginFactory
//----------------------------------------------------------------------------
IFactoryPtr Factory::detectPluginFactory(const yat::URI& uri) 
{
  CDMA_STATIC_FUNCTION_TRACE("Factory::detectPluginFactory");
  IFactoryPtr result;
  IFactoryPtr tmp;
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
    
    // Ask if the URI is readable
    if( data_source->isReadable( uri ) )
    {
      // Ask if the plugin is the owner of that URI
      if( data_source->isProducer( uri ) )
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
    iterator++;
  }
  return result;
}

} // namespace cdma
