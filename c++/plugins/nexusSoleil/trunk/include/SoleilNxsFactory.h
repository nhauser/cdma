#ifndef __CDMA_SOLEIL_NXSFACTORY_H__
#define __CDMA_SOLEIL_NXSFACTORY_H__

#include <vector>
#include <string>
#include <list>

// YAT library
#include <yat/plugin/PlugInSymbols.h>
#include <yat/utils/URI.h>
#include <yat/plugin/IPlugInInfo.h>

// CDMA core
#include <cdma/exception/Exception.h>
#include <cdma/IFactory.h>
#include <cdma/dictionary/Key.h>

// NeXus engine
#include <NxsDataset.h>

namespace cdma
{
namespace soleil
{
namespace nexus
{

// Soleil NeXus plug-in
#define SOLEIL_NXS_FACTORY_NAME "Factory"

// Debug macro helper
#define FUNCTION_TRACE(x) CDMA_FUNCTION_TRACE(std::string("cdma::soleil::nexus::") + std::string(x))

//==============================================================================
/// Plugin info class
//==============================================================================
class FactoryInfo : public yat::IPlugInInfo
{
public:
  virtual std::string get_plugin_id(void) const;
  virtual std::string get_interface_name(void) const;
  virtual std::string get_version_number(void) const;
};

//==============================================================================
/// IFactory implementation
//==============================================================================
class Factory : public cdma::IFactory 
{
public:
  Factory();
  ~Factory();

  //@{ IFactory methods

  cdma::IDatasetPtr openDataset(const std::string& location) throw ( cdma::Exception );
  cdma::DictionaryPtr openDictionary(const std::string& filepath) throw ( cdma::Exception );
  cdma::IDatasetPtr createDatasetInstance(const std::string& uri) throw ( cdma::Exception );
  cdma::IDatasetPtr createEmptyDatasetInstance() throw ( cdma::Exception );
  std::string getPathSeparator();
  std::string getName() { return SOLEIL_NXS_FACTORY_NAME; };
  cdma::IDataSourcePtr getPluginURIDetector();
  std::list<std::string> getPluginMethodsList();

  //@} IFactory methods

  inline static std::string plugin_id() { return "SoleilNeXus"; }
  inline static std::string interface_name() { return "cdma::IFactory"; }
  inline static std::string version_number() { return "1.0.0"; }

};

} //namespace nexus
} //namespace soleil
} //namespace cdma

#endif //__CDMA_IFACTORY_H__

