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
#include <cdma/Factory.h>
#include <cdma/IFactory.h>
#include <cdma/dictionary/Key.h>

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

  cdma::LogicalGroupPtr getLogicalRoot();
  
  //@}

private:

  // Constructor
  Dataset(const yat::URI& location, Factory* factory_ptr);
  Dataset();

};

} //namespace nexus
} //namespace soleil
} //namespace cdma

#endif //__CDMA_IFACTORY_H__

