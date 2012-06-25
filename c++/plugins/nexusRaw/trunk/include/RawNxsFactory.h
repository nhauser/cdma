#ifndef __CDMA_RAW_NXSFACTORY_H__
#define __CDMA_RAW_NXSFACTORY_H__

#include <vector>
#include <string>
#include <list>

// YAT library
#include <yat/plugin/PlugInSymbols.h>
#include <yat/plugin/IPlugInInfo.h>
#include <yat/utils/URI.h>

// CDMA core
#include <cdma/exception/Exception.h>
#include <cdma/IFactory.h>
#include <cdma/dictionary/Key.h>

// NeXus engine
#include <NxsDataset.h>

// Raw NeXus plug-in
#include <RawNxsFactory.h>

namespace cdma
{

namespace soleil
{

namespace rawnexus
{

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

  IDatasetPtr openDataset( const std::string& location ) throw ( cdma::Exception );
  DictionaryPtr openDictionary( const std::string& filepath ) throw ( cdma::Exception );
  IDatasetPtr createDatasetInstance( const std::string& uri ) throw ( cdma::Exception );
  IDatasetPtr createEmptyDatasetInstance() throw ( cdma::Exception );
  std::string getPathSeparator();
  std::string getName() { return NXS_FACTORY_NAME; };
  IDataSourcePtr getPluginURIDetector();
  std::list<std::string> getPluginMethodsList();

  //@} IFactory methods

  inline static std::string plugin_id() { return "RawNeXus"; }
  inline static std::string interface_name() { return "cdma::IFactory"; }
  inline static std::string version_number() { return "1.0.0"; }
};

//==============================================================================
/// Dataset class based on the NeXus engine implementation
/// See cdma::IDataset definition for more explanations
//==============================================================================
class Dataset : public cdma::nexus::Dataset
{
friend class Factory;

public:

  //@{ IDataset methods

   cdma::LogicalGroupPtr getLogicalRoot();
  
  //@}

private:

  // Constructor
  Dataset( const yat::URI& location, Factory* factory_ptr );
  Dataset();

};

//==============================================================================
/// IDataSource implementation
//==============================================================================
class DataSource : public cdma::IDataSource 
{
friend class Factory;

public:
  DataSource()  {};
  ~DataSource() {};

  //@{ IDataSource methods ------------

  bool isReadable(const yat::URI& destination) const;
  bool isBrowsable(const yat::URI& destination) const;
  bool isProducer(const yat::URI& destination) const;
  bool isExperiment(const yat::URI& destination) const;
  
  //@}

private:
  DataSource(Factory *factory_ptr): m_factory_ptr(factory_ptr) {};

  Factory *m_factory_ptr;
};

} // namespace rawnexus
} // namespace soleil
} // namespace cdma

#endif //__CDMA_RAW_NXSFACTORY_H__

