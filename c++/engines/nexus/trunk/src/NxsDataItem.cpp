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

namespace cdma
{
namespace nexus
{

//=============================================================================
//
// DataItem
//
//=============================================================================
//---------------------------------------------------------------------------
// DataItem::DataItem
//---------------------------------------------------------------------------
DataItem::DataItem(Dataset* dataset_ptr, const std::string& path)
{
  CDMA_FUNCTION_TRACE("cdma::nexus::DataItem::DataItem");
  init(dataset_ptr, path);
}
DataItem::DataItem(Dataset* dataset_ptr, const IGroupPtr& parent, const std::string& name )
{
  CDMA_FUNCTION_TRACE("cdma::nexus::DataItem::DataItem");
  init( dataset_ptr, parent->getLocation() + "/" + name );
}
DataItem::DataItem(Dataset* dataset_ptr, const NexusDataSetInfo& item, const std::string& path)
{
  CDMA_FUNCTION_TRACE("cdma::nexus::DataItem::DataItem");
  init( dataset_ptr, yat::String(path) );
  m_item = item;
}

//---------------------------------------------------------------------------
// DataItem::~DataItem
//---------------------------------------------------------------------------
DataItem::~DataItem()
{
  CDMA_TRACE("cdma::nexus::DataItem::~DataItem");
};

//---------------------------------------------------------------------------
// DataItem::init
//---------------------------------------------------------------------------
void DataItem::init(Dataset* dataset_ptr, const std::string& path, bool init_from_file)
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
// DataItem::findAttributeIgnoreCase
//---------------------------------------------------------------------------
cdma::IAttributePtr DataItem::findAttributeIgnoreCase(const std::string& attrName)
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
// DataItem::findDimensionView
//---------------------------------------------------------------------------
int DataItem::findDimensionView(const std::string& dimName)
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
// DataItem::getParent
//---------------------------------------------------------------------------
cdma::IGroupPtr DataItem::getParent()
{
  if( m_dataset_ptr )
    return m_dataset_ptr->getGroupFromPath( m_path );
  THROW_INVALID_POINTER("Pointer to Dataset object is no longer valid", "cdma::nexus::DataItem::getParent");
}

//---------------------------------------------------------------------------
// DataItem::getRoot
//---------------------------------------------------------------------------
cdma::IGroupPtr DataItem::getRoot()
{
  if( m_dataset_ptr )
    return m_dataset_ptr->getRootGroup();
  THROW_INVALID_POINTER("Pointer to Dataset object is no longer valid", "cdma::nexus::DataItem::getParent");
}

//---------------------------------------------------------------------------
// DataItem::getData
//---------------------------------------------------------------------------
cdma::ArrayPtr DataItem::getData(std::vector<int> position) throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("cdma::nexus::DataItem::getData(vector<int> position)");
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
// DataItem::getData
//---------------------------------------------------------------------------
cdma::ArrayPtr DataItem::getData(std::vector<int> origin, std::vector<int> shape) throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("cdma::nexus::DataItem::getData(vector<int> origin, vector<int> shape)");

  checkArray();
  std::vector<int> stride = m_array_ptr->getView()->getStride();

  cdma::ViewPtr view = new cdma::View( shape, origin, stride );
  cdma::ArrayPtr array_ptr = new cdma::Array( *m_array_ptr, view );
  return array_ptr;
}

//---------------------------------------------------------------------------
// DataItem::getDescription
//---------------------------------------------------------------------------
std::string DataItem::getDescription()
{
  if( hasAttribute("description") )
  {
    return getAttribute("description")->getStringValue();
  }
  return yat::String::nil;
}

//---------------------------------------------------------------------------
// DataItem::getDimensions
//---------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> DataItem::getDimensions(int order)
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
// DataItem::getDimensionList
//---------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> DataItem::getDimensionList()
{
  return getParent()->getDimensionList();
}

//---------------------------------------------------------------------------
// DataItem::getDimensionsString
//---------------------------------------------------------------------------
std::string DataItem::getDimensionsString()
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
// DataItem::getElementSize
//---------------------------------------------------------------------------
int DataItem::getElementSize()
{
  return m_item.DatumSize();
}

//---------------------------------------------------------------------------
// DataItem::getRank
//---------------------------------------------------------------------------
int DataItem::getRank()
{
  return m_item.Rank();
}

//---------------------------------------------------------------------------
// DataItem::getShape
//---------------------------------------------------------------------------
std::vector<int> DataItem::getShape()
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
// DataItem::getSize
//---------------------------------------------------------------------------
long DataItem::getSize()
{
  return m_item.Size();
}

//---------------------------------------------------------------------------
// DataItem::getSlice
//---------------------------------------------------------------------------
cdma::IDataItemPtr DataItem::getSlice(int dim, int value) throw ( cdma::Exception )
{
  int* shape = m_item.TotalDimArray();
  int* start = m_item.StartArray();
  int rank = m_item.Rank();

  // Check requested values are not out of range
  if( dim >= getRank() || value >= shape[dim] )
  {
    THROW_INVALID_RANGE("Requested slice is out of range!", "cdma::nexus::DataItem::getSlice" );
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
  DataItem* dataItem = new DataItem( m_dataset_ptr, *nexusItem, m_path);

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
// DataItem::getType
//---------------------------------------------------------------------------
const std::type_info& DataItem::getType()
{
  return TypeUtils::toCType(m_item.DataType());
}

//---------------------------------------------------------------------------
// DataItem::getUnitsString
//---------------------------------------------------------------------------
std::string DataItem::getUnitsString()
{
  if( hasAttribute("units") )
    return getAttribute("units")->getStringValue();
  return yat::String::nil;
}

//---------------------------------------------------------------------------
// DataItem::isMemberOfStructure
//---------------------------------------------------------------------------
bool DataItem::isMemberOfStructure()
{
  return false;
}

//---------------------------------------------------------------------------
// DataItem::isMetadata
//---------------------------------------------------------------------------
bool DataItem::isMetadata()
{
  return hasAttribute("signal");
}

//---------------------------------------------------------------------------
// DataItem::isScalar
//---------------------------------------------------------------------------
bool DataItem::isScalar()
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
// DataItem::isUnlimited
//---------------------------------------------------------------------------
bool DataItem::isUnlimited()
{
  // For now, that shouldn't happen until a while
  return false;
}

//---------------------------------------------------------------------------
// DataItem::isUnsigned
//---------------------------------------------------------------------------
bool DataItem::isUnsigned()
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
// DataItem::readScalarByte
//---------------------------------------------------------------------------
unsigned char DataItem::readScalarByte() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<unsigned char>();
}

//---------------------------------------------------------------------------
// DataItem::readScalarDouble
//---------------------------------------------------------------------------
double DataItem::readScalarDouble() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<double>();
}

//---------------------------------------------------------------------------
// DataItem::readScalarFloat
//---------------------------------------------------------------------------
float DataItem::readScalarFloat() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<float>();
}

//---------------------------------------------------------------------------
// DataItem::readScalarInt
//---------------------------------------------------------------------------
int DataItem::readScalarInt() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<int>();
}

//---------------------------------------------------------------------------
// DataItem::readScalarLong
//---------------------------------------------------------------------------
long DataItem::readScalarLong() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<long>();
}

//---------------------------------------------------------------------------
// DataItem::readScalarShort
//---------------------------------------------------------------------------
short DataItem::readScalarShort() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<short>();
}

//---------------------------------------------------------------------------
// DataItem::readString
//---------------------------------------------------------------------------
std::string DataItem::readString() throw ( cdma::Exception )
{
  checkArray();
  return m_array_ptr->getValue<std::string>();
}

//---------------------------------------------------------------------------
// DataItem::removeAttribute
//---------------------------------------------------------------------------
bool DataItem::removeAttribute(const cdma::IAttributePtr& attr)
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
// DataItem::setDataType
//---------------------------------------------------------------------------
void DataItem::setDataType(const std::type_info& type)
{
  NexusDataType dataType = TypeUtils::toNexusDataType( type );
  m_item.SetInfo( dataType, m_item.Rank() );
}

//---------------------------------------------------------------------------
// DataItem::setData
//---------------------------------------------------------------------------
void DataItem::setData(const cdma::ArrayPtr& array)
{
  m_array_ptr = array;
}

//---------------------------------------------------------------------------
// DataItem::setDimensions
//---------------------------------------------------------------------------
void DataItem::setDimensions(const std::string&)
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::DataItem::setDimensions");
}

//---------------------------------------------------------------------------
// DataItem::setDimension
//---------------------------------------------------------------------------
void DataItem::setDimension(const cdma::IDimensionPtr& dim, int order) throw ( cdma::Exception )
{
  IGroupPtr parent = getParent();
  
  if( parent )
  {
    dim->setDisplayOrder( order );
    parent->addDimension(dim);
  }
}

//---------------------------------------------------------------------------
// DataItem::setUnitsString
//---------------------------------------------------------------------------
void DataItem::setUnitsString(const std::string& unit)
{
  Attribute* attr = new Attribute( );
  attr->setName("units");
  attr->setStringValue(unit);
  addAttribute( attr );
}

//---------------------------------------------------------------------------
// DataItem::addAttribute
//---------------------------------------------------------------------------
void DataItem::addAttribute(const cdma::IAttributePtr& attr)
{
/*
  Attribute* attr = new Attribute();
  IAttributePtr attribute = attr;
  attr->setName( name );
  attr->setValue( value );
  m_attr_map[name] = attribute;
*/
  m_attr_map[attr->getName()] = attr;  
//  return attribute;
}

//---------------------------------------------------------------------------
// DataItem::getAttribute
//---------------------------------------------------------------------------
IAttributePtr DataItem::getAttribute( const std::string& attr_name )
{
  AttributeMap::iterator it = m_attr_map.find(attr_name);
  if( it != m_attr_map.end() )
    return it->second;

  return IAttributePtr(NULL);
}

//---------------------------------------------------------------------------
// DataItem::getAttributeList
//---------------------------------------------------------------------------
AttributeList DataItem::getAttributeList()
{
  AttributeList attr_list;
  for( AttributeMap::iterator it = m_attr_map.begin(); it != m_attr_map.end(); it++ )
    attr_list.push_back(it->second);
  return attr_list;
}

//---------------------------------------------------------------------------
// DataItem::getLocation
//---------------------------------------------------------------------------
std::string DataItem::getLocation() const
{
  return Dataset::concatPath(m_path, m_nodeName);
}

//---------------------------------------------------------------------------
// DataItem::setLocation
//---------------------------------------------------------------------------
void DataItem::setLocation(const std::string& path)
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
// DataItem::getName
//---------------------------------------------------------------------------
std::string DataItem::getName() const
{
  return m_name;
}

//---------------------------------------------------------------------------
// DataItem::getShortName
//---------------------------------------------------------------------------
std::string DataItem::getShortName() const
{
  return m_shortName;
}

//---------------------------------------------------------------------------
// DataItem::hasAttribute
//---------------------------------------------------------------------------
bool DataItem::hasAttribute( const std::string& attr_name )
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
// DataItem::setName
//---------------------------------------------------------------------------
void DataItem::setName(const std::string& name)
{
  m_name = name;
}

//---------------------------------------------------------------------------
// DataItem::setShortName
//---------------------------------------------------------------------------
void DataItem::setShortName(const std::string& name)
{
  m_shortName = name;
}

//---------------------------------------------------------------------------
// DataItem::setParent
//---------------------------------------------------------------------------
void DataItem::setParent(const cdma::IGroupPtr&)
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::DataItem::setParent");
}

//---------------------------------------------------------------------------
// DataItem::checkArray
//---------------------------------------------------------------------------
void DataItem::checkArray()
{
  if( !m_array_ptr )
  {
    loadArray();
  }
}

//---------------------------------------------------------------------------
// DataItem::loadArray
//---------------------------------------------------------------------------
void DataItem::loadArray()
{
  CDMA_FUNCTION_TRACE("cdma::nexus::DataItem::loadArray");
  
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
    cdma::Array *array_ptr = new cdma::Array( TypeUtils::toRawCType(m_item.DataType()),
                                              shape, data.Data() );

    // remove ownership of the NexusDataSet
    data.SetData(NULL);

    // update Array
    m_array_ptr.reset(array_ptr);
  }
  else
  {
    THROW_FILE_ACCESS("Unable to read data: file is closed!", "cdma::nexus::DataItem::loadArray");
  }
}

//---------------------------------------------------------------------------
// DataItem::initAttr
//---------------------------------------------------------------------------
void DataItem::initAttr()
{
  CDMA_FUNCTION_TRACE("cdma::nexus::DataItem::initAttr");
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
      // Create Attribute
      m_attr_map[attr_info.AttrName()] = new Attribute( file, attr_info );
    }
  }
}

//---------------------------------------------------------------------------
// DataItem::open
//---------------------------------------------------------------------------
void DataItem::open( bool openNode )
{
  CDMA_FUNCTION_TRACE("cdma::nexus::DataItem::open");
  
  // Open parent node
  NexusFilePtr file = m_dataset_ptr->getHandle();
  
  if( file->CurrentGroupPath() != m_path || file->CurrentDataset() != m_nodeName )
  {
    if( file->CurrentGroupPath() != m_path )
    {
      if( ! file->OpenGroupPath(m_path.data() ) )
      {
        THROW_FILE_ACCESS("Unable to open path!\nPath: " + m_path, "cdma::nexus::DataItem::DataItem");
      }
    }
    
    // Open item node
    if( openNode && file->CurrentDataset() != m_nodeName)
    {
      if( ! file->OpenDataSet( m_nodeName.data(), false ) )
      {
        THROW_FILE_ACCESS("Unable to open node!\nName: " +  m_shortName, "cdma::nexus::DataItem::DataItem");
      }
    }
  }
}

} // namespace nexus
} // namespace cdma
