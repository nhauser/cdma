//*****************************************************************************
/// Synchrotron SOLEIL
///
/// Recording configuration parsor
///
/// Creation : 15/04/2005
/// Author   : S. Poirier
///
//*****************************************************************************

#include <yat/utils/String.h>
#include <yat/utils/Logging.h>
#include <yat/file/FileName.h>
#include <cdma/Factory.h>
#include <cdma/dictionary/Path.h>
#include <cdma/IFactory.h>

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
// Factory::init
//----------------------------------------------------------------------------
void Factory::init(const std::string &plugin_path)
{
  yat::FileEnum fe(plugin_path + '/', yat::FileEnum::ENUM_FILE);
  std::cout << "Factory::init " << fe.path() << std::endl;

  if( !fe.path_exist() )
    throw cdma::Exception("BAD_PATH", PSZ_FMT("Path not exists (%s)", PSZ(plugin_path)), "Factory::init");
  
  while( fe.find() )
  {
    std::cout << "Factory::init found " << fe.name_ext() << " " << SHARED_LIB_EXTENSION << std::endl;
    if( fe.ext().is_equal(SHARED_LIB_EXTENSION) )
    {
      //- we found a shared lib
      Plugin plugin_objects;

      std::cout << "Factory::init found lib " << fe.full_name() << std::endl;
      try
      {
        PluginInfoPair plugin_pair = instance().m_plugin_manager.load( fe.full_name() );
        plugin_objects.info    = plugin_pair.first;
        plugin_objects.factory = plugin_pair.second;
      std::cout << "Factory::init lib loaded" << std::endl;
      }
      catch( yat::Exception& e)
      {
        e.dump();
        continue;
      }
      std::cout << "Factory::init plugin_objects.info->get_interface_name: " << plugin_objects.info->get_interface_name() << std::endl;
      
      if ( plugin_objects.info->get_interface_name() == INTERFACE_NAME )
      {
        // We found a CDMA plugin!
        std::cout << "Factory::init: We found a CDMA plugin!" << std::endl;
        yat::IPlugInInfo* plugin_info = 0;
        try
        {
          plugin_info = static_cast<yat::IPlugInInfo*>(plugin_objects.info);
          if (plugin_info == NULL)
            throw std::bad_cast();
          
          std::string plugin_id = plugin_objects.info->get_plugin_id();
          std::cout << "Factory::init plugin_objects.info->get_plugin_id: " << plugin_objects.info->get_plugin_id() << std::endl;
          //yat::log_info("cdma", PSZ_FMT("Loading plugin %s", plugin_id));
          
          std::cout << "Factory::init plugin before registered" << std::endl;
          instance().m_plugin_map[plugin_id] = plugin_objects;
          std::cout << "Factory::init plugin registered" << std::endl;
        }
        catch( std::bad_cast& )
        {
          //- this is a CDMA, but its info class is not a yat::IPlugInInfo
          //- ==> invalid plugin
          yat::log_error("cdma", PSZ_FMT("Invalid CDMA plugin : %s", fe.path()));
          continue;
        }
      }
    }
  }
  std::cout << "Factory::init Done" << std::endl;
}

//----------------------------------------------------------------------------
// Factory::setActiveView
//----------------------------------------------------------------------------
IFactoryPtr Factory::getPluginFactory(const std::string &plugin_id)
{
  PluginFactoryPtrMap::iterator it = instance().m_plugin_factory_map.find(plugin_id);
  if( it == instance().m_plugin_factory_map.end() )
  {  
    // Find the plugin
    PluginMap::iterator it = instance().m_plugin_map.find(plugin_id);
    if( it == instance().m_plugin_map.end() )
      throw cdma::Exception("NOT_FOUND", PSZ_FMT("No such plugin (%s).", plugin_id), "cdma::Factory::getPluginFactory");
    
    yat::IPlugInFactory* factory = (*it).second.factory;

    // Instanciate the IFactory implementation
    yat::IPlugInObject* obj;
    factory->create(obj);
    IFactoryPtr factory_ptr = static_cast<cdma::IFactory*>(obj);
    instance().m_plugin_factory_map[plugin_id] = factory_ptr;
    
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
IDatasetPtr Factory::openDataset(const std::string& uri) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::openDataset");
}

//----------------------------------------------------------------------------
// Factory::openDictionary
//----------------------------------------------------------------------------
DictionaryPtr Factory::openDictionary(const std::string& filepath) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("Factory::openDictionary");
}

//----------------------------------------------------------------------------
// Factory::createArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createArray(const std::type_info clazz, const std::vector<int> shape)
{
  THROW_NOT_IMPLEMENTED("Factory::createArray");
}

//----------------------------------------------------------------------------
// Factory::createArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createArray(const void * array)
{
  THROW_NOT_IMPLEMENTED("Factory::createArray");
}

//----------------------------------------------------------------------------
// Factory::createStringArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createStringArray(const std::string& value)
{
  THROW_NOT_IMPLEMENTED("Factory::createStringArray");
}

//----------------------------------------------------------------------------
// Factory::createDoubleArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createDoubleArray(double array[])
{
  THROW_NOT_IMPLEMENTED("Factory::createDoubleArray");
}

//----------------------------------------------------------------------------
// Factory::createDoubleArray
//----------------------------------------------------------------------------
ArrayPtr Factory::createDoubleArray(double array[], const std::vector<int> shape)
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
KeyPtr Factory::createKey(std::string keyName)
{
  THROW_NOT_IMPLEMENTED("Factory::createKey");
}

//----------------------------------------------------------------------------
// Factory::createPath
//----------------------------------------------------------------------------
PathPtr Factory::createPath( std::string path )
{
  THROW_NOT_IMPLEMENTED("Factory::createPath");
}

//----------------------------------------------------------------------------
// Factory::createPathParameter
//----------------------------------------------------------------------------
PathParameterPtr Factory::createPathParameter(CDMAType::ParameterType type, std::string& name, void * value)
{
  THROW_NOT_IMPLEMENTED("Factory::createPathParameter");
}

//----------------------------------------------------------------------------
// Factory::createPathParamResolver
//----------------------------------------------------------------------------
IPathParamResolverPtr Factory::createPathParamResolver(const PathPtr& path)
{
  THROW_NOT_IMPLEMENTED("Factory::createPathParamResolver");
}

} // namespace cdma
