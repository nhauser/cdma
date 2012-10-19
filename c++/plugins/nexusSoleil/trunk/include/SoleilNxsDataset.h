#ifndef __CDMA_SOLEIL_NXSDATASET_H__
#define __CDMA_SOLEIL_NXSDATASET_H__

#include <vector>
#include <string>

// YAT library
#include <yat/plugin/PlugInSymbols.h>
#include <yat/utils/URI.h>
#include <yat/plugin/IPlugInInfo.h>

// CDMA core
#include <cdma/exception/Exception.h>
#include <cdma/factory/Factory.h>
#include <cdma/factory/plugin/IPluginFactory.h>
#include <cdma/dictionary/IKey.h>

// NeXus engine
#include <NxsDataset.h>

// Soleil NeXus plug-in
#include <SoleilNxsFactory.h>
#include <DictionaryDetector.h>

namespace cdma
{
namespace soleil
{
namespace nexus
{

//==============================================================================
/// Dataset class based on the NeXus engine implementation
/// See cdma::IDataset definition for more explanations
//==============================================================================
class Dataset : public cdma::nexus::Dataset
{
friend class Factory;

public:

  //@{ IDataset methods

  cdma::ILogicalGroupPtr getLogicalRoot();
  IDataItemPtr getItemFromPath(const std::string &fullPath);
  IDataItemPtr getItemFromPath(const yat::String &path, const yat::String& name);
  IContainerPtr findContainerByPath(const std::string& path);
  
  //@}

private:

  // Constructor
  Dataset(const yat::URI& location, Factory* factory_ptr);
  Dataset();

  /// Return the root path of the dataset inside the file defined by the fragment part of the URI
  ///
  std::string getRootPath() const;

};

} //namespace nexus
} //namespace soleil
} //namespace cdma

#endif //__CDMA_IFACTORY_H__

