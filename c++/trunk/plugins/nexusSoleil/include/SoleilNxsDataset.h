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
private:
  DictionaryDetector    m_detector;     ///< Mapping file name for that is used for the dictionary

  /// Retrieve the correct mapping file according to structure of the Nexus file
  ///
  const std::string& getMappingFileName();

public:

  /// Constructor
  ///  param :
  /// @param : filepath string representing the file path
  ///
  SoleilNxsDataset(const std::string& filepath);

  /// Return the the logical root of the dataset.
  ///
  /// @return shared pointer on the LogicalGroup
  ///
  LogicalGroupPtr getLogicalRoot();

};

} //namespace CDMACore
#endif //__CDMA_IFACTORY_H__

