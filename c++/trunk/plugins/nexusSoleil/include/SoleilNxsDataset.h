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
/// Dataset implementation based on the NeXus plug-in implementation
  /// See cdma::IDataset definition for more explanation
//==============================================================================
class SoleilNxsDataset : public NxsDataset
{
public:

  /// Create an instance on an existing dataset
  ///
  /// @param : filepath string representing the file path
  ///
  static NxsDatasetPtr getDataset(const yat::URI& location, SoleilNxsFactory *factory_ptr);

  /// Create an instance on an new dataset object
  ///
  static NxsDatasetPtr newDataset();

  //@{ IDataset methods

  LogicalGroupPtr getLogicalRoot();
  
  //@}

private:

  // Constructor
  SoleilNxsDataset(const yat::URI& location);
  SoleilNxsDataset();

  SoleilNxsFactory* m_factory_ptr; // C-style pointer on factory. Never delete it !!
};

} //namespace CDMACore
#endif //__CDMA_IFACTORY_H__

