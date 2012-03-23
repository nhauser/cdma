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
#include <TypeUtils.h>
#include <cdma/utils/ArrayUtils.h>

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
NxsDataItem::NxsDataItem(NxsDataset* dataset_ptr, const std::string& path)
{
  CDMA_FUNCTION_TRACE("NxsDataItem::NxsDataItem");
  init(dataset_ptr, path);
}
NxsDataItem::NxsDataItem(NxsDataset* dataset_ptr, const IGroupPtr& parent, const std::string& name )
{
  CDMA_FUNCTION_TRACE("NxsDataItem::NxsDataItem");
  init( dataset_ptr, parent->getLocation() + "/" + name );
}
NxsDataItem::NxsDataItem(NxsDataset* dataset_ptr, const NexusDataSetInfo& item, const std::string& path)
{
  CDMA_FUNCTION_TRACE("NxsDataItem::NxsDataItem");
  init( dataset_ptr, yat::String(path) );
  m_item = item;
}

//---------------------------------------------------------------------------
// NxsDataItem::~NxsDataItem
//---------------------------------------------------------------------------
NxsDataItem::~NxsDataItem()
{
  CDMA_TRACE("NxsDataItem::~NxsDataItem");
};

//---------------------------------------------------------------------------
// NxsDataItem::init
//---------------------------------------------------------------------------
void NxsDataItem::init(NxsDataset* dataset_ptr, const std::string& path, bool init_from_file)
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
  m_nodeName = nodes[nodes.size() - 1 ];
  m_shortName = m_nodeName;
  m_name = m_nodeName;
  m_dataset_ptr = dataset_ptr;

  NexusFilePtr file = m_dataset_ptr->getHandle();
  if( init_from_file )
  {
    // Open path
    open(false);

    // Init attribute list
    initAttr();
    
    // Read node's info
    file->GetDataSetInfo( &m_item, m_nodeName.c_str() );

    // Close all nodes
    file->CloseAllGroups();
  }
  
  if( hasAttribute("long_name") )
  {
    m_name = getAttribute("long_name")->getStringValue();
  }
}

//---------------------------------------------------------------------------
// NxsDataItem::findAttributeIgnoreCase
//---------------------------------------------------------------------------
cdma::IAttributePtr NxsDataItem::findAttributeIgnoreCase(const std::string& attrName)
{
  IAttributePtr attr (NULL);
  yat::String referent (attrName);
  for( AttributeMap::iterator it = m_attr_map.begin(); it != m_attr_map.end(); it++ )
  {
    if( referent.is_equal_no_case(it->first) )
    {
      attr = it->second;
      break;
    }
  }
  return attr;
}

//---------------------------------------------------------------------------
// NxsDataItem::findDimensionView
//---------------------------------------------------------------------------
int NxsDataItem::findDimensionView(const std::string& dimName)
{
  int order = -1;
  IGroupPtr parent = getParent();
  if( parent )
  {
    std::list<cdma::IDimensionPtr> dimList = parent->getDimensionList();
    for( std::list<cdma::IDimensionPtr>::iterator it = dimList.begin(); it != dimList.end(); it++ )
    {
      if( (*it)->getName() == dimName )
      {
        order = (*it)->getDimensionAxis();
      }
    }
  }
  return order;
}

//---------------------------------------------------------------------------
// NxsDataItem::getParent
//---------------------------------------------------------------------------
cdma::IGroupPtr NxsDataItem::getParent()
{
  if( m_dataset_ptr )
    return m_dataset_ptr->getGroupFromPath( m_path );
  THROW_INVALID_POINTER("Pointer to Dataset object is no longer valid", "NxsDataItem::getParent");
}

//---------------------------------------------------------------------------
// NxsDataItem::getRoot
//---------------------------------------------------------------------------
cdma::IGroupPtr NxsDataItem::getRoot()
{
  if( m_dataset_ptr )
    return m_dataset_ptr->getRootGroup();
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
      shape.push_back( m_item.DimArray()[dim] - position[dim] );
    }
    else
    {
      origin.push_back( 0 );
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
  if( hasAttribute("description") )
  {
    return getAttribute("description")->getStringValue();
  }
  return yat::String::nil;
}

//---------------------------------------------------------------------------
// NxsDataItem::getDimensions
//---------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> NxsDataItem::getDimensions(int order)
{
  std::list<cdma::IDimensionPtr> result;
  IGroupPtr parent = getParent();
  if( parent )
  {
    std::list<cdma::IDimensionPtr> dimList = parent->getDimensionList();
    for( std::list<cdma::IDimensionPtr>::iterator it = dimList.begin(); it != dimList.end(); it++ )
    {
      if( order == (*it)->getDimensionAxis() )
      {
        result.push_back( *it );
      }
    }
  }
  return result;
}

//---------------------------------------------------------------------------
// NxsDataItem::getDimensionList
//---------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> NxsDataItem::getDimensionList()
{
  return getParent()->getDimensionList();
}

//---------------------------------------------------------------------------
// NxsDataItem::getDimensionsString
//---------------------------------------------------------------------------
std::string NxsDataItem::getDimensionsString()
{
  std::stringstream res;
  std::list<cdma::IDimensionPtr> dim_list = getDimensionList();
  for( std::list<cdma::IDimensionPtr>::iterator it = dim_list.begin(); it != dim_list.end(); it++ )
  {
    if( it != dim_list.begin() )
      res << " ";
    res << (*it)->getName();
  }
  return res.str();
}

//---------------------------------------------------------------------------
// NxsDataItem::getElementSize
//---------------------------------------------------------------------------
int NxsDataItem::getElementSize()
{
  return m_item.DatumSize();
}

//---------------------------------------------------------------------------
// NxsDataItem::getRank
//---------------------------------------------------------------------------
int NxsDataItem::getRank()
{
  return m_item.Rank();
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
// NxsDataItem::getSlice
//---------------------------------------------------------------------------
cdma::IDataItemPtr NxsDataItem::getSlice(int dim, int value) throw ( cdma::Exception )
{
  int* shape = m_item.TotalDimArray();
  int* start = m_item.StartArray();
  int rank = m_item.Rank();

  // Check requested values are not out of range
  if( dim >= getRank() || value >= shape[dim] )
  {
    THROW_INVALID_RANGE("Requested slice is out of range!", "NxsDataItem::getSlice" );
  } 

  cdma::IDataItemPtr result (NULL);
  cdma::ArrayPtr array = getData();

  // Search the non-reduced dimsension that given 'dim' corresponds to
  int realDim  = 0;

  // Calculate the new shape and start position
  rank = m_item.TotalRank();
  int* piStart = new int[rank];
  int* piDim   = new int[rank];
  for( int i = 0; i < rank; i++ )
  {
      // Respect the slab position of this dataitem
      if( start )
      {
        piStart[i] = start[i];
      }
      else
      {
        piStart[i] = 0;
      }
      piDim[i] = shape[i];
      
      // Increment realDim until it reaches the requested one (avoid reduced dimensions)
      if( shape[i] > 1 && realDim < dim )
      {
        realDim++;
      }
  }
  piDim[realDim]   = 1;
  piStart[realDim] = value;

  // Create the corresponding item
  NexusDataSet* nexusItem = new NexusDataSet(m_item.DataType(), NULL, rank, piDim, piStart);
  NxsDataItem* dataItem = new NxsDataItem( m_dataset_ptr, *nexusItem, m_path);

  if( array ) 
  {
    // Slice the loaded matrix
    ArrayUtils util ( array );
    array = util.slice( dim, value )->getArray();
    
    // Set the Array to the created IDataItem
    dataItem->m_array_ptr = array;
  }
  
  result = dataItem;
  
  return result;
}

//---------------------------------------------------------------------------
// NxsDataItem::getType
//---------------------------------------------------------------------------
const std::type_info& NxsDataItem::getType()
{
  return TypeUtils::toCType(m_item.DataType());
}

//---------------------------------------------------------------------------
// NxsDataItem::getUnitsString
//---------------------------------------------------------------------------
std::string NxsDataItem::getUnitsString()
{
  if( hasAttribute("units") )
    return getAttribute("units")->getStringValue();
  return yat::String::nil;
}

//---------------------------------------------------------------------------
// NxsDataItem::isMemberOfStructure
//---------------------------------------------------------------------------
bool NxsDataItem::isMemberOfStructure()
{
  return false;
}

//---------------------------------------------------------------------------
// NxsDataItem::isMetadata
//---------------------------------------------------------------------------
bool NxsDataItem::isMetadata()
{
  return hasAttribute("signal");
}

//---------------------------------------------------------------------------
// NxsDataItem::isScalar
//---------------------------------------------------------------------------
bool NxsDataItem::isScalar()
{
  if( m_shape.empty() || (m_shape.size() == 1 && m_shape[0] == 1) )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//---------------------------------------------------------------------------
// NxsDataItem::isUnlimited
//---------------------------------------------------------------------------
bool NxsDataItem::isUnlimited()
{
  // For now, that shouldn't happen until a while
  return false;
}

//---------------------------------------------------------------------------
// NxsDataItem::isUnsigned
//---------------------------------------------------------------------------
bool NxsDataItem::isUnsigned()
{
  switch( m_item.DataType() )
  {
    case NX_UINT8:
    case NX_UINT16:
    case NX_UINT32:
    case NX_UINT64:
      return true;
    default:
      return false;
  }
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
  THROW_NOT_IMPLEMENTED("NxsDataItem::readString");
}

//---------------------------------------------------------------------------
// NxsDataItem::removeAttribute
//---------------------------------------------------------------------------
bool NxsDataItem::removeAttribute(const cdma::IAttributePtr& attr)
{
  AttributeMap::iterator iter = m_attr_map.find( attr->getName() );
  if( iter != m_attr_map.end() )
  {
    m_attr_map.erase( iter );
    return true;
  }
  return false;
}

//---------------------------------------------------------------------------
// NxsDataItem::setDataType
//---------------------------------------------------------------------------
void NxsDataItem::setDataType(const std::type_info& type)
{
  NexusDataType dataType = TypeUtils::toNexusDataType( type );
  m_item.SetInfo( dataType, m_item.Rank() );
}

//---------------------------------------------------------------------------
// NxsDataItem::setData
//---------------------------------------------------------------------------
void NxsDataItem::setData(const cdma::ArrayPtr& array)
{
  m_array_ptr = array;
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
void NxsDataItem::setDimension(const cdma::IDimensionPtr& dim, int order) throw ( cdma::Exception )
{
  IGroupPtr parent = getParent();
  
  if( parent )
  {
    dim->setDisplayOrder( order );
    parent->addDimension(dim);
  }
}

//---------------------------------------------------------------------------
// NxsDataItem::setUnitsString
//---------------------------------------------------------------------------
void NxsDataItem::setUnitsString(const std::string& unit)
{
  NxsAttribute* attr = new NxsAttribute( );
  attr->setName("units");
  attr->setStringValue(unit);
  addAttribute( attr );
}

//---------------------------------------------------------------------------
// NxsDataItem::addAttribute
//---------------------------------------------------------------------------
void NxsDataItem::addAttribute(const cdma::IAttributePtr& attr)
{
/*
  NxsAttribute* attr = new NxsAttribute();
  IAttributePtr attribute = attr;
  attr->setName( name );
  attr->setValue( value );
  m_attr_map[name] = attribute;
*/
  m_attr_map[attr->getName()] = attr;  
//  return attribute;
}

//---------------------------------------------------------------------------
// NxsDataItem::getAttribute
//---------------------------------------------------------------------------
IAttributePtr NxsDataItem::getAttribute( const std::string& attr_name )
{
  AttributeMap::iterator it = m_attr_map.find(attr_name);
  if( it != m_attr_map.end() )
    return it->second;

  return IAttributePtr(NULL);
}

//---------------------------------------------------------------------------
// NxsDataItem::getAttributeList
//---------------------------------------------------------------------------
AttributeList NxsDataItem::getAttributeList()
{
  AttributeList attr_list;
  for( AttributeMap::iterator it = m_attr_map.begin(); it != m_attr_map.end(); it++ )
    attr_list.push_back(it->second);
  return attr_list;
}

//---------------------------------------------------------------------------
// NxsDataItem::getLocation
//---------------------------------------------------------------------------
std::string NxsDataItem::getLocation() const
{
  return NxsDataset::concatPath(m_path, m_nodeName);
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
  m_nodeName = item;
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
bool NxsDataItem::hasAttribute( const std::string& attr_name )
{
  if( m_attr_map.find(attr_name) != m_attr_map.end() )
  {
    return true;
  }
  else
  {
    return false;
  }
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
  
  if( m_dataset_ptr )
  {
    NexusFilePtr file_ptr = m_dataset_ptr->getHandle();
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
    file_ptr->GetData( &data, m_nodeName.data() );
  
    // Init Array
    cdma::Array *array_ptr = new cdma::Array( TypeUtils::toCType(m_item.DataType()),
                                              shape, data.Data() );

    // remove ownership of the NexusDataSet
    data.SetData(NULL);

    // update Array
    m_array_ptr.reset(array_ptr);
  }
  else
  {
    THROW_FILE_ACCESS("Unable to read data: file is closed!", "NxsDataItem::loadArray");
  }
}

//---------------------------------------------------------------------------
// NxsDataItem::initAttr
//---------------------------------------------------------------------------
void NxsDataItem::initAttr()
{
  CDMA_FUNCTION_TRACE("NxsDataItem::initAttr");
  // Clear previously loaded attr
  m_attr_map.clear();
  
  // Open node
  open();
  
  // Load attributes
  NexusFilePtr file = m_dataset_ptr->getHandle();
  if( file->AttrCount() > 0 )
  {
    CDMA_TRACE("attr count: " << file->AttrCount());
    // Read attr infos
    NexusAttrInfo attr_info;
    
    // Iterate over attributes collection
    for( int rc = file->GetFirstAttribute(&attr_info); 
         NX_OK == rc; 
         rc = file->GetNextAttribute(&attr_info) )
    {
      // Create NxsAttribute
      m_attr_map[attr_info.AttrName()] = new NxsAttribute( file, attr_info );
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
  NexusFilePtr file = m_dataset_ptr->getHandle();
  
  if( file->CurrentGroupPath() != m_path || file->CurrentDataset() != m_nodeName )
  {
    if( file->CurrentGroupPath() != m_path )
    {
      if( ! file->OpenGroupPath(m_path.data() ) )
      {
        THROW_FILE_ACCESS("Unable to open path!\nPath: " + m_path, "NxsDataItem::NxsDataItem");
      }
    }
    
    // Open item node
    if( openNode && file->CurrentDataset() != m_nodeName)
    {
      if( ! file->OpenDataSet( m_nodeName.data(), false ) )
      {
        THROW_FILE_ACCESS("Unable to open node!\nName: " +  m_shortName, "NxsDataItem::NxsDataItem");
      }
    }
  }
}

} // namespace
