// ****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez Clément
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
#include <cdma/Common.h>
#include <NxsDataItem.h>
#include <TypeDetector.h>
#define TEMP_EXCEPTION(a,b) throw cdma::Exception("TARTAMPION", a, b)

/// A DataItem is a logical container for data. It has a DataType, a set of
/// Dimensions that define its array shape, and optionally a set of Attributes.
namespace cdma
{

//=============================================================================
//
// NxsDataItem
//
//=============================================================================
//---------------------------------------------------------------------------
// NxsDataItem::NxsDataItem
//---------------------------------------------------------------------------
NxsDataItem::NxsDataItem(NxsDatasetWPtr dataset_wptr, const std::string& path)
{
  CDMA_FUNCTION_TRACE("NxsDataItem::NxsDataItem");
  init(dataset_wptr, path);
}
NxsDataItem::NxsDataItem(NxsDatasetWPtr dataset_wptr, const IGroupPtr& parent, const std::string& name )
{
  CDMA_FUNCTION_TRACE("NxsDataItem::NxsDataItem");
  init( dataset_wptr, parent->getLocation() + "/" + name );
}
NxsDataItem::NxsDataItem(NxsDatasetWPtr dataset_wptr, const NexusDataSetInfo& item, const std::string& path)
{
  CDMA_FUNCTION_TRACE("NxsDataItem::NxsDataItem");
  init( dataset_wptr, yat::String(path), false );
  m_item = item;
}

//---------------------------------------------------------------------------
// NxsDataItem::~NxsDataItem
//---------------------------------------------------------------------------
NxsDataItem::~NxsDataItem()
{
};

//---------------------------------------------------------------------------
// NxsDataItem::init
//---------------------------------------------------------------------------
void NxsDataItem::init(NxsDatasetWPtr dataset_wptr, const std::string& path, bool init_from_file)
{
  // Resolve dataitem name and path
  std::vector<yat::String> nodes;
  // First the path
  yat::String tmp (path);
  tmp.split('/', &nodes);
  tmp = "/";
  for( yat::uint16 i = 0; i < nodes.size() - 1; i++ )
  {
    if( !nodes[i].empty() )
    {
      tmp += nodes[i] + "/";
    }
  }
  m_path = tmp;

  // Second the node name
  yat::String item = nodes[nodes.size() - 1 ];
  m_name = item;

  m_dataset_wptr = dataset_wptr;

  NexusFilePtr file = m_dataset_wptr.lock()->getHandle();
  if( init_from_file )
  {
    // Open path
    open(false);

    // Init attribute list
    initAttr();
    
    // Read node's info
    file->GetDataSetInfo( &m_item, m_name.c_str() );

    // Close all nodes
    file->CloseAllGroups();
  }
}

//---------------------------------------------------------------------------
// NxsDataItem::findAttributeIgnoreCase
//---------------------------------------------------------------------------
cdma::IAttributePtr NxsDataItem::findAttributeIgnoreCase(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::findAttributeIgnoreCase");
}

//---------------------------------------------------------------------------
// NxsDataItem::findDimensionView
//---------------------------------------------------------------------------
int NxsDataItem::findDimensionView(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::findDimensionView");
}

//---------------------------------------------------------------------------
// NxsDataItem::getASlice
//---------------------------------------------------------------------------
cdma::IDataItemPtr NxsDataItem::getASlice(int /*dimension*/, int /*value*/) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getASlice");
}

//---------------------------------------------------------------------------
// NxsDataItem::getParent
//---------------------------------------------------------------------------
cdma::IGroupPtr NxsDataItem::getParent()
{
  if( m_dataset_wptr )
    return m_dataset_wptr.lock()->getGroupFromPath( m_path );
  THROW_INVALID_POINTER("Pointer to Dataset object is no longer valid", "NxsDataItem::getParent");
}

//---------------------------------------------------------------------------
// NxsDataItem::getRoot
//---------------------------------------------------------------------------
cdma::IGroupPtr NxsDataItem::getRoot()
{
  if( m_dataset_wptr )
    return m_dataset_wptr.lock()->getRootGroup();
  THROW_INVALID_POINTER("Pointer to Dataset object is no longer valid", "NxsDataItem::getParent");
}

//---------------------------------------------------------------------------
// NxsDataItem::getData
//---------------------------------------------------------------------------
cdma::ArrayPtr NxsDataItem::getData(std::vector<int> position) throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("NxsDataItem::getData(vector<int> position)");
  int node_rank = m_item.Rank();
  
  std::vector<int> origin;
  std::vector<int> shape;

  for( int dim = 0; dim < node_rank; dim++ )
  {
    if( dim < (int)(position.size()) )
    {
      origin.push_back( position[dim] );
      CDMA_TRACE("shape.push_back " << m_item.DimArray()[dim] - position[dim]);
      shape.push_back( m_item.DimArray()[dim] - position[dim] );
    }
    else
    {
      origin.push_back( 0 );
      CDMA_TRACE("shape.push_back " << m_item.DimArray()[dim]);
      shape.push_back( m_item.DimArray()[dim] );
    }
  }
  cdma::ArrayPtr array = getData(origin, shape);
  return array;
}

//---------------------------------------------------------------------------
// NxsDataItem::getData
//---------------------------------------------------------------------------
cdma::ArrayPtr NxsDataItem::getData(std::vector<int> origin, std::vector<int> shape) throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("NxsDataItem::getData(vector<int> origin, vector<int> shape)");

  checkArray();
  int rank = m_array_ptr->getRank();
  int* iShape = new int[rank];
  int* iStart = new int[rank];
  for( int i = 0; i < rank; i++ )
  {
    iStart[i] = origin[i];
    iShape[i]  = shape[i];
  }
  cdma::ViewPtr view = new cdma::View( rank, iShape, iStart );
  cdma::ArrayPtr array_ptr = new cdma::Array( *m_array_ptr, view );
  return array_ptr;
}

//---------------------------------------------------------------------------
// NxsDataItem::getDescription
//---------------------------------------------------------------------------
std::string NxsDataItem::getDescription()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getDescription");
}

//---------------------------------------------------------------------------
// NxsDataItem::getDimensions
//---------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> NxsDataItem::getDimensions(int)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getDimensions");
}

//---------------------------------------------------------------------------
// NxsDataItem::getDimensionList
//---------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> NxsDataItem::getDimensionList()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getDimensionList");
}

//---------------------------------------------------------------------------
// NxsDataItem::getDimensionsString
//---------------------------------------------------------------------------
std::string NxsDataItem::getDimensionsString()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getDimensionsString");
}

//---------------------------------------------------------------------------
// NxsDataItem::getElementSize
//---------------------------------------------------------------------------
int NxsDataItem::getElementSize()
{
  return m_item.DatumSize();
}

//---------------------------------------------------------------------------
// NxsDataItem::getNameAndDimensions
//---------------------------------------------------------------------------
std::string NxsDataItem::getNameAndDimensions()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getNameAndDimensions");
}

//---------------------------------------------------------------------------
// NxsDataItem::getNameAndDimensions
//---------------------------------------------------------------------------
std::string NxsDataItem::getNameAndDimensions(bool /*useFullName*/, bool /*showDimLength*/)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getNameAndDimensions");
}

//---------------------------------------------------------------------------
// NxsDataItem::getRangeList
//---------------------------------------------------------------------------
std::list<cdma::RangePtr> NxsDataItem::getRangeList()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getRangeList");
}

//---------------------------------------------------------------------------
// NxsDataItem::getRank
//---------------------------------------------------------------------------
int NxsDataItem::getRank()
{
  return m_item.Rank();
}

//---------------------------------------------------------------------------
// NxsDataItem::getSection
//---------------------------------------------------------------------------
cdma::IDataItemPtr NxsDataItem::getSection(std::list<cdma::RangePtr> /*section*/) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getSection");
}

//---------------------------------------------------------------------------
// NxsDataItem::getSectionRanges
//---------------------------------------------------------------------------
std::list<cdma::RangePtr> NxsDataItem::getSectionRanges()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getSectionRanges");
}

//---------------------------------------------------------------------------
// NxsDataItem::getShape
//---------------------------------------------------------------------------
std::vector<int> NxsDataItem::getShape()
{
  if( m_shape.empty() )
  {
    int* shape = m_item.DimArray();
    for( int i = 0; i < m_item.Rank(); i++ )
    {
      m_shape.push_back( shape[i] );
    }
  }
  return m_shape;
}

//---------------------------------------------------------------------------
// NxsDataItem::getSize
//---------------------------------------------------------------------------
long NxsDataItem::getSize()
{
  return m_item.Size();
}

//---------------------------------------------------------------------------
// NxsDataItem::getSizeToCache
//---------------------------------------------------------------------------
int NxsDataItem::getSizeToCache()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getSizeToCache");
}

//---------------------------------------------------------------------------
// NxsDataItem::getSlice
//---------------------------------------------------------------------------
cdma::IDataItemPtr NxsDataItem::getSlice(int /*dim*/, int /*value*/) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getSlice");
}

//---------------------------------------------------------------------------
// NxsDataItem::getType
//---------------------------------------------------------------------------
const std::type_info& NxsDataItem::getType()
{
  return TypeDetector::detectType(m_item.DataType());
}

//---------------------------------------------------------------------------
// NxsDataItem::getUnitsString
//---------------------------------------------------------------------------
std::string NxsDataItem::getUnitsString()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getUnitsString");
}

//---------------------------------------------------------------------------
// NxsDataItem::hasCachedData
//---------------------------------------------------------------------------
bool NxsDataItem::hasCachedData()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::hasCachedData");
}

//---------------------------------------------------------------------------
// NxsDataItem::findAttributeIgnoreCase
//---------------------------------------------------------------------------
//##int NxsDataItem::hashCode()

//---------------------------------------------------------------------------
// NxsDataItem::invalidateCache
//---------------------------------------------------------------------------
void NxsDataItem::invalidateCache()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::invalidateCache");
}

//---------------------------------------------------------------------------
// NxsDataItem::isCaching
//---------------------------------------------------------------------------
bool NxsDataItem::isCaching()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::isCaching");
}

//---------------------------------------------------------------------------
// NxsDataItem::isMemberOfStructure
//---------------------------------------------------------------------------
bool NxsDataItem::isMemberOfStructure()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::isMemberOfStructure");
}

//---------------------------------------------------------------------------
// NxsDataItem::isMetadata
//---------------------------------------------------------------------------
bool NxsDataItem::isMetadata()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::isMetadata");
}

//---------------------------------------------------------------------------
// NxsDataItem::isScalar
//---------------------------------------------------------------------------
bool NxsDataItem::isScalar()
{
  if( m_shape.empty() || (m_shape.size() == 1 && m_shape[0] == 1) )
    return true;

  return false;
}

//---------------------------------------------------------------------------
// NxsDataItem::isUnlimited
//---------------------------------------------------------------------------
bool NxsDataItem::isUnlimited()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::isUnlimited");
}

//---------------------------------------------------------------------------
// NxsDataItem::isUnsigned
//---------------------------------------------------------------------------
bool NxsDataItem::isUnsigned()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::isUnsigned");
}

//---------------------------------------------------------------------------
// NxsDataItem::readScalarByte
//---------------------------------------------------------------------------
unsigned char NxsDataItem::readScalarByte() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<unsigned char>();
}

//---------------------------------------------------------------------------
// NxsDataItem::readScalarDouble
//---------------------------------------------------------------------------
double NxsDataItem::readScalarDouble() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<double>();
}

//---------------------------------------------------------------------------
// NxsDataItem::readScalarFloat
//---------------------------------------------------------------------------
float NxsDataItem::readScalarFloat() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<float>();
}

//---------------------------------------------------------------------------
// NxsDataItem::readScalarInt
//---------------------------------------------------------------------------
int NxsDataItem::readScalarInt() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<int>();
}

//---------------------------------------------------------------------------
// NxsDataItem::readScalarLong
//---------------------------------------------------------------------------
long NxsDataItem::readScalarLong() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<long>();
}

//---------------------------------------------------------------------------
// NxsDataItem::readScalarShort
//---------------------------------------------------------------------------
short NxsDataItem::readScalarShort() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<short>();
}

//---------------------------------------------------------------------------
// NxsDataItem::readString
//---------------------------------------------------------------------------
std::string NxsDataItem::readString() throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::readScalarString");
}

//---------------------------------------------------------------------------
// NxsDataItem::removeAttribute
//---------------------------------------------------------------------------
bool NxsDataItem::removeAttribute(const cdma::IAttributePtr&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::removeAttribute");
}

//---------------------------------------------------------------------------
// NxsDataItem::setCachedData
//---------------------------------------------------------------------------
//##void NxsDataItem::setCachedData(Array& cacheData, bool isMetadata) throw ( cdma::Exception )

//---------------------------------------------------------------------------
// NxsDataItem::setCaching
//---------------------------------------------------------------------------
void NxsDataItem::setCaching(bool)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::setCaching");
}

//---------------------------------------------------------------------------
// NxsDataItem::setDataType
//---------------------------------------------------------------------------
void NxsDataItem::setDataType(const std::type_info&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::setDataType");
}

//---------------------------------------------------------------------------
// NxsDataItem::setDimensions
//---------------------------------------------------------------------------
void NxsDataItem::setDimensions(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::setDimensions");
}

//---------------------------------------------------------------------------
// NxsDataItem::setDimension
//---------------------------------------------------------------------------
void NxsDataItem::setDimension(const cdma::IDimensionPtr&, int) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::setDimension");
}

//---------------------------------------------------------------------------
// NxsDataItem::setElementSize
//---------------------------------------------------------------------------
void NxsDataItem::setElementSize(int)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::setElementSize");
}

//---------------------------------------------------------------------------
// NxsDataItem::setSizeToCache
//---------------------------------------------------------------------------
void NxsDataItem::setSizeToCache(int)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::setSizeToCache");
}

//---------------------------------------------------------------------------
// NxsDataItem::setUnitsString
//---------------------------------------------------------------------------
void NxsDataItem::setUnitsString(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::setUnitsString");
}

//---------------------------------------------------------------------------
// NxsDataItem::clone
//---------------------------------------------------------------------------
IDataItemPtr NxsDataItem::clone()
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::clone");
}

//---------------------------------------------------------------------------
// NxsDataItem::addAttribute
//---------------------------------------------------------------------------
cdma::IAttributePtr NxsDataItem::addAttribute(const std::string&, yat::Any&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::addOneAttribute");
}

//---------------------------------------------------------------------------
// NxsDataItem::getAttribute
//---------------------------------------------------------------------------
IAttributePtr NxsDataItem::getAttribute(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::getAttribute");
}

//---------------------------------------------------------------------------
// NxsDataItem::getAttributeList
//---------------------------------------------------------------------------
std::list<yat::SharedPtr<cdma::IAttribute, yat::Mutex> > NxsDataItem::getAttributeList()
{
  return m_attr_list;
}

//---------------------------------------------------------------------------
// NxsDataItem::getLocation
//---------------------------------------------------------------------------
std::string NxsDataItem::getLocation() const
{
  return NxsDataset::concatPath(m_path, m_name);
}

//---------------------------------------------------------------------------
// NxsDataItem::setLocation
//---------------------------------------------------------------------------
void NxsDataItem::setLocation(const std::string& path)
{
  std::vector<yat::String> nodes;
  yat::String tmp (path);
  tmp.split('/', &nodes);
  yat::String item = nodes[nodes.size() - 1 ];
  tmp = "/";
  for( yat::uint16 i = 0; i < nodes.size() - 1; i++ )
  {
    if( !nodes[i].empty() )
    {
      tmp += nodes[i] + "/";
    }
  }
  m_path = tmp;
  m_name = item;
}

//---------------------------------------------------------------------------
// NxsDataItem::getName
//---------------------------------------------------------------------------
std::string NxsDataItem::getName() const
{
  return m_name;
}

//---------------------------------------------------------------------------
// NxsDataItem::getShortName
//---------------------------------------------------------------------------
std::string NxsDataItem::getShortName() const
{
  return m_shortName;
}

//---------------------------------------------------------------------------
// NxsDataItem::hasAttribute
//---------------------------------------------------------------------------
bool NxsDataItem::hasAttribute(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::hasAttribute");
}

//---------------------------------------------------------------------------
// NxsDataItem::setName
//---------------------------------------------------------------------------
void NxsDataItem::setName(const std::string& name)
{
  m_name = name;
}

//---------------------------------------------------------------------------
// NxsDataItem::setShortName
//---------------------------------------------------------------------------
void NxsDataItem::setShortName(const std::string& name)
{
  m_shortName = name;
}

//---------------------------------------------------------------------------
// NxsDataItem::setParent
//---------------------------------------------------------------------------
void NxsDataItem::setParent(const cdma::IGroupPtr&)
{
  THROW_NOT_IMPLEMENTED("NxsDataItem::setParent");
}

//---------------------------------------------------------------------------
// NxsDataItem::getDataset
//---------------------------------------------------------------------------
cdma::IDatasetPtr NxsDataItem::getDataset()
{
  return m_dataset_wptr.lock();
}

//---------------------------------------------------------------------------
// NxsDataItem::checkArray
//---------------------------------------------------------------------------
void NxsDataItem::checkArray()
{
  if( !m_array_ptr )
  {
    loadArray();
  }
}

//---------------------------------------------------------------------------
// NxsDataItem::loadArray
//---------------------------------------------------------------------------
void NxsDataItem::loadArray()
{
  CDMA_FUNCTION_TRACE("NxsDataItem::loadArray");
  
  NxsDatasetPtr dataset_ptr = m_dataset_wptr.lock();

  if( dataset_ptr )
  {
    NexusFilePtr file_ptr = dataset_ptr->getHandle();
    NexusFileAccess auto_open(file_ptr);

    // prepare shape
    std::vector<int> shape;
    for( int i = 0; i < m_item.Rank(); i++ )
    {
      shape.push_back(m_item.DimArray()[i]);
    }

    // Open parent group
    open(false);

    // Init nexusdataset
    NexusDataSet data;

    // Load data
    file_ptr->GetData( &data, m_name.data() );
  
    // Init Array
    cdma::Array *array_ptr = new cdma::Array( TypeDetector::detectType(m_item.DataType()),
                                              shape, data.Data() );

    // remove ownership of the NexusDataSet
    data.SetData(NULL);

    // update Array
    m_array_ptr.reset(array_ptr);
  }
  else
  {
    TEMP_EXCEPTION("Unable to read data: file is closed!", "NxsDataItem::loadArray");
  }
}

//---------------------------------------------------------------------------
// NxsDataItem::initAttr
//---------------------------------------------------------------------------
void NxsDataItem::initAttr()
{
  // Clear previously loaded attr
  if( !m_attr_list.empty() )
  {
    m_attr_list.clear();
  }
  
  // Open node
  open();
  
  // Load attributes
  NexusFilePtr file = m_dataset_wptr.lock()->getHandle();
  int nbAttr = file->AttrCount();
  if( nbAttr > 0 )
  {
    // Read attr infos
    yat::String path = m_path + "/" + m_name;
    NexusAttrInfo* tmp_attr = new NexusAttrInfo();
    NxsAttribute *attr;
    
    // Iterate over nexus attr
    file->GetFirstAttribute(tmp_attr);
    attr = new NxsAttribute( file, tmp_attr );
    m_attr_list.push_back( IAttributePtr(attr) );
    for( int i = 0; i < nbAttr - 1; i++ )
    {
      // Scan attribute
      tmp_attr = new NexusAttrInfo();
      file->GetNextAttribute(tmp_attr);
      
      // Create NxsAttribute
      attr = new NxsAttribute( file, tmp_attr );
      m_attr_list.push_back( IAttributePtr(attr) );
    }
  }
}

//---------------------------------------------------------------------------
// NxsDataItem::open
//---------------------------------------------------------------------------
void NxsDataItem::open( bool openNode )
{
  CDMA_FUNCTION_TRACE("NxsDataItem::open");
  
  // Open parent node
  NexusFilePtr file = m_dataset_wptr.lock()->getHandle();
  
  if( file->CurrentGroupPath() != m_path || file->CurrentDataset() != m_name )
  {
    if( file->CurrentGroupPath() != m_path )
    {
      if( ! file->OpenGroupPath(m_path.data() ) )
      {
        TEMP_EXCEPTION("Unable to open path!\nPath: " + m_path, "NxsDataItem::NxsDataItem");
      }
    }
    
    // Open item node
    if( openNode && file->CurrentDataset() != m_name)
    {
      if( ! file->OpenDataSet( m_name.data(), false ) )
      {
        TEMP_EXCEPTION("Unable to open node!\nName: " +  m_name, "NxsDataItem::NxsDataItem");
      }
    }
  }
}

} // namespace
