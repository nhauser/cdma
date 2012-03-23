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

// Soleil NeXus plug-in
#define SOLEIL_NXS_FACTORY_NAME "SoleilNxsFactory"

//==============================================================================
/// Plugin info class
//==============================================================================
class SoleilNxsFactoryInfo : public yat::IPlugInInfo
{
public:
  virtual std::string get_plugin_id(void) const;
  virtual std::string get_interface_name(void) const;
  virtual std::string get_version_number(void) const;
};

//==============================================================================
/// IFactory implementation
//==============================================================================
class SoleilNxsFactory : public cdma::IFactory 
{
public:
  SoleilNxsFactory();
  ~SoleilNxsFactory();

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
} //namespace soleil_nexus
#endif //__CDMA_IFACTORY_H__

