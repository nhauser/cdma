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
#ifndef __CDMA_NXSDATAITEM_H__
#define __CDMA_NXSDATAITEM_H__

#ifndef __TYPEINFO_INCLUDED__
 #include <typeinfo>
 #define __TYPEINFO_INCLUDED__
#endif

// Tools lib
#include <list>
#include <vector>
#include <yat/utils/String.h>

// CDMA Core
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/array/impl/Array.h>

// Plug-in
#include <internal/common.h>
#include <NxsAttribute.h>
#include <NxsGroup.h>

namespace cdma
{

class CDMA_DECL NxsDataItem : public IDataItem
{
private:
  std::list<IAttributePtr> m_attr_list;
  NxsDataset*              m_dataset_ptr;
  yat::String              m_name;        // Name of the dataitem (ie: attribute long_name else node's name)
  yat::String              m_shortName;
  yat::String              m_path;        // Path of the item through the dataset file structure (excluding item node name)
  NexusDataSetInfo         m_item;        // Info on the belonged data
  IArrayPtr                m_array;       // IArray object
  std::vector<int>         m_shape;       // Shape defined by the NexusDatasetInfo

public:
  // Constructors
  NxsDataItem(NxsDataset* dataset, const char* path, bool init_from_file = true );
  NxsDataItem(NxsDataset* dataset, const IGroupPtr& parent, const char* name );
  NxsDataItem(NxsDataset* dataset, const NexusDataSetInfo& item, const std::string& path);
  ~NxsDataItem() { CDMA_FUNCTION_TRACE("NxsDataItem::~NxsDataItem"); };

  //@{ IDataItem interface
  IAttributePtr findAttributeIgnoreCase(const std::string& name);
  int findDimensionIndex(const std::string& name);
  IDataItemPtr getASlice(int dimension, int value) throw ( cdma::Exception );
  IGroupPtr getParent();
  IGroupPtr getRoot();
  IArrayPtr getData(std::vector<int> position = std::vector<int>() ) throw ( cdma::Exception );
  IArrayPtr getData(std::vector<int> origin, std::vector<int> shape) throw ( cdma::Exception );
  std::string getDescription();
  std::list<IDimensionPtr> getDimensions(int i);
  std::list<IDimensionPtr> getDimensionList();
  std::string getDimensionsString();
  int getElementSize();
  std::string getNameAndDimensions();
  std::string getNameAndDimensions(bool useFullName, bool showDimLength);
  std::list<IRangePtr> getRangeList();
  int getRank();
  IDataItemPtr getSection(std::list<IRangePtr> section) throw ( cdma::Exception );
  std::list<IRangePtr> getSectionRanges();
  std::vector<int> getShape();
  long getSize();
  int getSizeToCache();
  IDataItemPtr getSlice(int dim, int value) throw ( cdma::Exception );
  const std::type_info& getType();
  std::string getUnitsString();
  bool hasCachedData();
  void invalidateCache();
  bool isCaching();
  bool isMemberOfStructure();
  bool isMetadata();
  bool isScalar();
  bool isUnlimited();
  bool isUnsigned();
  unsigned char readScalarByte() throw ( cdma::Exception );
  double readScalarDouble() throw ( cdma::Exception );
  float readScalarFloat() throw ( cdma::Exception );
  int readScalarInt() throw ( cdma::Exception );
  long readScalarLong() throw ( cdma::Exception );
  short readScalarShort() throw ( cdma::Exception );
  std::string readScalarString() throw ( cdma::Exception );
  void setCaching(bool caching);
  void setDataType(const std::type_info& dataType);
  void setDimensions(const std::string& dimString);
  void setDimension(const IDimensionPtr& dim, int ind) throw ( cdma::Exception );
  void setElementSize(int elementSize);
  void setSizeToCache(int sizeToCache);
  void setUnitsString(const std::string& units);
  IDataItemPtr clone();
  void addOneAttribute(const IAttributePtr&);
  void addStringAttribute(const std::string&, const std::string&);
  IAttributePtr getAttribute(const std::string&);
  std::list<IAttributePtr > getAttributeList();
  std::string getLocation();
  std::string getName();
  std::string getShortName();
  bool hasAttribute(const std::string&, const std::string&);
  bool removeAttribute(const IAttributePtr&);
  void setName(const std::string&);
  void setShortName(const std::string&);
  void setParent(const IGroupPtr&);
  IDatasetPtr getDataset();
  
  //@}
  //@{IObject interface
  CDMAType::ModelType getModelType() const { return CDMAType::DataItem; };
  std::string getFactoryName() const { return NXS_FACTORY_NAME; };
  //@}
  //@{plugin methods
  void setLocation(const std::string& path);
  
protected:
  void init(NxsDataset* dataset, std::string path, bool init_from_file = true);
private:
  void loadMatrix();
  void initAttr();
  void open(bool openNode = true );
  //@}
};

typedef yat::SharedPtr<NxsDataItem, yat::Mutex> NxsDataItemPtr;
}
#endif
