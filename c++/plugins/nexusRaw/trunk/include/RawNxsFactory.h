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

//==============================================================================
/// Plugin info class
//==============================================================================
class CDMA_DECL RawNxsFactoryInfo : public yat::IPlugInInfo
{
public:
  virtual std::string get_plugin_id(void) const;
  virtual std::string get_interface_name(void) const;
  virtual std::string get_version_number(void) const;
};

//==============================================================================
/// IFactory implementation
//==============================================================================
class CDMA_DECL RawNxsFactory : public IFactory 
{
public:
  RawNxsFactory();
  ~RawNxsFactory();

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
class CDMA_DECL RawNxsDataset : public NxsDataset
{
friend class RawNxsFactory;

public:

  //@{ IDataset methods

   LogicalGroupPtr getLogicalRoot();
  
  //@}

private:

  // Constructor
  RawNxsDataset( const yat::URI& location, RawNxsFactory* factory_ptr );
  RawNxsDataset();

};

//==============================================================================
/// IDataSource implementation
//==============================================================================
class RawNxsDataSource : public IDataSource 
{
friend class RawNxsFactory;

public:
  RawNxsDataSource()  {};
  ~RawNxsDataSource() {};

  //@{ IDataSource methods ------------

  bool isReadable(const yat::URI& destination) const;
  bool isBrowsable(const yat::URI& destination) const;
  bool isProducer(const yat::URI& destination) const;
  bool isExperiment(const yat::URI& destination) const;
  
  //@}

private:
  RawNxsDataSource(RawNxsFactory *factory_ptr): m_factory_ptr(factory_ptr) {};

  RawNxsFactory *m_factory_ptr;
};

} //namespace cdma

#endif //__CDMA_RAW_NXSFACTORY_H__

