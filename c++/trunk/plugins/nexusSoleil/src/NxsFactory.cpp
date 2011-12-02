// ****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : See AUTHORS file
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// ****************************************************************************

#include <yat/plugin/PlugInSymbols.h>

#include <NxsFactory.h>
#include <NxsDataset.h>
#include <cdma/dictionary/Key.h>

EXPORT_SINGLECLASS_PLUGIN(cdma::NxsFactory, \
                          cdma::NxsFactoryInfo);

namespace cdma
{

const std::string PlugInID         ( "SoleilNeXus" );
const std::string InterfaceName    ( "cdma::IFactory" );
const std::string VersionNumber    ( "1.0.0" );

//----------------------------------------------------------------------------
// NxsFactoryInfo::get_plugin_id
//----------------------------------------------------------------------------
std::string NxsFactoryInfo::get_plugin_id() const
{
  return PlugInID;
}

//----------------------------------------------------------------------------
// NxsFactoryInfo::get_interface_name
//----------------------------------------------------------------------------
std::string NxsFactoryInfo::get_interface_name() const
{
  return InterfaceName;
}

//----------------------------------------------------------------------------
// NxsFactoryInfo::get_version_number
//----------------------------------------------------------------------------
std::string NxsFactoryInfo::get_version_number() const
{
  return VersionNumber;
}

//==============================================================================
// class NxsFactory
//==============================================================================
//----------------------------------------------------------------------------
// NxsFactory::openDataset
//----------------------------------------------------------------------------
IDatasetPtr NxsFactory::openDataset(const std::string& uri) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsFactory::openDataset");
}

//----------------------------------------------------------------------------
// NxsFactory::openDictionary
//----------------------------------------------------------------------------
DictionaryPtr NxsFactory::openDictionary(const std::string& filepath) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsFactory::openDictionary");
}

//----------------------------------------------------------------------------
// NxsFactory::createArray
//----------------------------------------------------------------------------
IArrayPtr NxsFactory::createArray(const std::type_info clazz, const std::vector<int> shape)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createArray");
}

//----------------------------------------------------------------------------
// NxsFactory::createArray
//----------------------------------------------------------------------------
IArrayPtr NxsFactory::createArray(const std::type_info clazz, const std::vector<int> shape, const void * storage)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createArray");
}

//----------------------------------------------------------------------------
// NxsFactory::createArray
//----------------------------------------------------------------------------
IArrayPtr NxsFactory::createArray(const void * array)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createArray");
}

//----------------------------------------------------------------------------
// NxsFactory::createStringArray
//----------------------------------------------------------------------------
IArrayPtr NxsFactory::createStringArray(const std::string& value)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createStringArray");
}

//----------------------------------------------------------------------------
// NxsFactory::createDoubleArray
//----------------------------------------------------------------------------
IArrayPtr NxsFactory::createDoubleArray(double array[])
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createDoubleArray");
}

//----------------------------------------------------------------------------
// NxsFactory::createDoubleArray
//----------------------------------------------------------------------------
IArrayPtr NxsFactory::createDoubleArray(double array[], const std::vector<int> shape)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createDoubleArray");
}

//----------------------------------------------------------------------------
// NxsFactory::createArrayNoCopy
//----------------------------------------------------------------------------
IArrayPtr NxsFactory::createArrayNoCopy(const void * array)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createArrayNoCopy");
}

//----------------------------------------------------------------------------
// NxsFactory::createDataItem
//----------------------------------------------------------------------------
IDataItemPtr NxsFactory::createDataItem(const cdma::IGroupPtr& parent, const std::string& shortName, const cdma::IArrayPtr& array) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createDataItem");
}

//----------------------------------------------------------------------------
// NxsFactory::createGroup
//----------------------------------------------------------------------------
IGroupPtr NxsFactory::createGroup(const cdma::IGroupPtr& parent, const std::string& shortName, const bool updateParent)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createGroup");
}

//----------------------------------------------------------------------------
// NxsFactory::createGroup
//----------------------------------------------------------------------------
IGroupPtr NxsFactory::createGroup(const std::string& shortName) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createGroup");
}

//----------------------------------------------------------------------------
// NxsFactory::createLogicalGroup
//----------------------------------------------------------------------------
LogicalGroupPtr NxsFactory::createLogicalGroup(cdma::IDataset* dataset, const cdma::KeyPtr& key)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createLogicalGroup");
}

//----------------------------------------------------------------------------
// NxsFactory::createAttribute
//----------------------------------------------------------------------------
IAttributePtr NxsFactory::createAttribute(const std::string& name, const void * value)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createAttribute");
}

//----------------------------------------------------------------------------
// NxsFactory::createDatasetInstance
//----------------------------------------------------------------------------
IDatasetPtr NxsFactory::createDatasetInstance(const std::string& uri) throw ( cdma::Exception )
{
  return new NxsDataset(uri);
}

//----------------------------------------------------------------------------
// NxsFactory::createEmptyDatasetInstance
//----------------------------------------------------------------------------
IDatasetPtr NxsFactory::createEmptyDatasetInstance() throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createEmptyDatasetInstance");
}

//----------------------------------------------------------------------------
// NxsFactory::createKey
//----------------------------------------------------------------------------
KeyPtr NxsFactory::createKey(std::string keyName)
{
  return new cdma::Key(keyName);
}

//----------------------------------------------------------------------------
// NxsFactory::createPath
//----------------------------------------------------------------------------
PathPtr NxsFactory::createPath( std::string path )
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createPath");
}

//----------------------------------------------------------------------------
// NxsFactory::createPathParameter
//----------------------------------------------------------------------------
PathParameterPtr NxsFactory::createPathParameter(cdma::CDMAType::ParameterType type, const std::string& name, void * value)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createPathParameter");
}

//----------------------------------------------------------------------------
// NxsFactory::createPathParamResolver
//----------------------------------------------------------------------------
IPathParamResolverPtr NxsFactory::createPathParamResolver(const cdma::PathPtr& path)
{
  THROW_NOT_IMPLEMENTED("NxsFactory::createPathParamResolver");
}

//----------------------------------------------------------------------------
// NxsFactory::getPathSeparator
//----------------------------------------------------------------------------
std::string NxsFactory::getPathSeparator()
{
  THROW_NOT_IMPLEMENTED("NxsFactory::getPathSeparator");
}

} // namespace cdma
