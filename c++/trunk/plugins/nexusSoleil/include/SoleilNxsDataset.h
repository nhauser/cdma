#ifndef __CDMA_SOLEIL_NXSDATASET_H__
#define __CDMA_SOLEIL_NXSDATASET_H__

#include <vector>
#include <string>

// YAT library
#include <yat/plugin/PlugInSymbols.h>
#include <yat/utils/URI.h>
#include <yat/plugin/IPlugInInfo.h>

// CDMA core
#include <cdma/IObject.h>
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

//==============================================================================
/// Dataset class based on the NeXus engine implementation
/// See cdma::IDataset definition for more explanations
//==============================================================================
class SoleilNxsDataset : public NxsDataset
{
friend class SoleilNxsFactory;

public:

  //@{ IDataset methods

    LogicalGroupPtr getLogicalRoot();
  
  //@}

private:

  // Constructor
  SoleilNxsDataset(const yat::URI& location, SoleilNxsFactory* factory_ptr);
  SoleilNxsDataset();

};

} //namespace cdma

#endif //__CDMA_IFACTORY_H__

