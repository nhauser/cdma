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
#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>
#include <cdma/IFactory.h>
#include <cdma/dictionary/Key.h>

// NeXus engine
#include <NxsDataset.h>

// Soleil NeXus plug-in
#include <SoleilNxsFactory.h>

namespace cdma
{
extern const std::string PlugInID;
extern const std::string InterfaceName;
extern const std::string VersionNumber;

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
class SoleilNxsFactory : public IFactory 
{
public:
  SoleilNxsFactory();
  ~SoleilNxsFactory();

  //@{ IFactory methods

  IDatasetPtr openDataset(const std::string& location) throw ( cdma::Exception );
  DictionaryPtr openDictionary(const std::string& filepath) throw ( cdma::Exception );
  IDatasetPtr createDatasetInstance(const std::string& uri) throw ( cdma::Exception );
  IDatasetPtr createEmptyDatasetInstance() throw ( cdma::Exception );
  std::string getPathSeparator();
  std::string getName() { return NXS_FACTORY_NAME; };
  IDataSourcePtr getPluginURIDetector();
  std::list<std::string> getPluginMethodsList();

  //@} IFactory methods

};
} //namespace CDMACore
#endif //__CDMA_IFACTORY_H__

