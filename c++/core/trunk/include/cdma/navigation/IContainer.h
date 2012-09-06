//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
//
// This file is part of cdma-core library.
//
// The cdma-core library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
//
// The CDMA library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
//
// Contributors :
// See AUTHORS file 
//******************************************************************************

#ifndef __CDMA_ICONTAINER_H__
#define __CDMA_ICONTAINER_H__

#include <cdma/Common.h>
#include <cdma/array/Range.h>
#include <cdma/navigation/IAttribute.h>

namespace cdma
{
// Forward declarations
DECLARE_CLASS_SHARED_PTR(IDataItem);
DECLARE_CLASS_SHARED_PTR(IGroup);
DECLARE_CLASS_SHARED_PTR(IContainer);
DECLARE_CLASS_SHARED_WEAK_PTR(IDataset);
DECLARE_CLASS_SHARED_PTR(LogicalGroup);

typedef std::list<IContainerPtr> IContainerPtrList;

//==============================================================================
/// @brief Base interface of IGroup and IDataItem
///
/// Must be implemented by concretized IGroup and IDataItem classes in data format engines
/// and, if needed, in plug-ins
//==============================================================================
class CDMA_DECL IContainer
{
public:

  /// Type
  ///
  /// Concrete container type
  ///
  enum Type
  {
    DATA_GROUP = 0,
    DATA_ITEM = 1
  };

  /// d-tor
  virtual ~IContainer()
  {
  }
  
  /// Add a new attribute to this Container.
  ///
  /// @param attribute  Attribute
  ///
  //virtual IAttributePtr addAttribute(const std::string& name, yat::Any &value) = 0;
  virtual void addAttribute(const IAttributePtr& attribute) = 0;
  
  /// Find an Attribute in this Group by its name.
  ///
  /// @param name  attribute's name
  /// @return the attribute, or null if not found
  ///
  virtual IAttributePtr getAttribute(const std::string& name) = 0;
  
  /// Get the set of attributes contained directly in this Group.
  ///
  /// @return list of type Attribute; may be empty, not null.
  ///
  virtual AttributeList getAttributeList() = 0;
  
  /// Get the location referenced in the Dataset.
  ///
  /// @return string type
  ///
  virtual std::string getLocation() const = 0;
  
  /// Get the (long) name of the IContainer.
  ///
  /// @return string type object
  ///
  virtual std::string getName() const = 0;
  
  /// Get the "short" name, unique within its parent Group.
  ///
  virtual std::string getShortName() const = 0;
  
  /// Check if the Group has an Attribute with certain name and value.
  ///
  virtual bool hasAttribute(const std::string& name) = 0;
  
  /// Remove an Attribute from the Attribute list.
  ///
  virtual bool removeAttribute(const IAttributePtr& attribute) = 0;
  
  /// Set the object's (long) name.
  ///
  /// @param name object's name
  ///
  virtual void setName(const std::string& name) = 0;
  
  /// Set the object's (short) name.
  ///
  /// @param name short name
  ///
  virtual void setShortName(const std::string& name) = 0;
  
  /// Return the real concrete container type
  ///
  virtual Type getContainerType() const = 0;

  /// Set the parent group.
  ///
  /// @param group
  ///      IGroup object
  ///
  /// virtual void setParent(const IGroupPtr& group) = 0;
};

} //namespace CDMACore
#endif //__CDMA_ICONTAINER_H__
