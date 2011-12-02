//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_ICONTAINER_H__
#define __CDMA_ICONTAINER_H__

#include <list>
#include <string>

#include <cdma/IObject.h>

namespace cdma
{
  
//==============================================================================
/// IContainer
/// Shared interface between Groups and DataItems.
//==============================================================================
class IContainer
{
public:
  virtual ~IContainer()
  {
  }
  
  /// Add an Attribute to the Group.
  ///
  /// @param attribute  Attribute
  ///
  virtual void addOneAttribute(const IAttributePtr& attribute) = 0;
  
  /// A convenience method of adding a string type attribute.
  ///
  /// @param name
  ///      string type object
  /// @param value
  ///      string type object Created on 06/03/2008
  ///
  virtual void addStringAttribute(const std::string& name, const std::string& value) = 0;
  
  /// Find an Attribute in this Group by its name.
  ///
  /// @param name
  ///      the name of the attribute
  /// @return the attribute, or null if not found
  ///
  virtual IAttributePtr getAttribute(const std::string& name) = 0;
  
  /// Get the set of attributes contained directly in this Group.
  ///
  /// @return list of type Attribute; may be empty, not null.
  ///
  virtual std::list<IAttributePtr> getAttributeList() = 0;
  
  /// Get the location referenced by the Dataset.
  ///
  /// @return string type Created on 18/06/2008
  ///
  virtual std::string getLocation() = 0;
  
  /// Get the (long) name of the IObject, which contains the path information.
  ///
  /// @return string type object Created on 18/06/2008
  ///
  virtual std::string getName() = 0;
  
  /// Get the "short" name, unique within its parent Group.
  ///
  virtual std::string getShortName() = 0;
  
  /// Check if the Group has an Attribute with certain name and value.
  ///
  virtual bool hasAttribute(const std::string& name, const std::string& value) = 0;
  
  /// Remove an Attribute from the Attribute list.
  ///
  virtual bool removeAttribute(const IAttributePtr& attribute) = 0;
  
  /// Set the IObject's (long) name.
  ///
  /// @param name
  ///      string object
  ///
  virtual void setName(const std::string& name) = 0;
  
  /// Set the IObject's (short) name.
  ///
  /// @param name
  ///      in string type Created on 18/06/2008
  ///
  virtual void setShortName(const std::string& name) = 0;
  
  /// Set the parent group.
  ///
  /// @param group
  ///      IGroup object
  ///
  /// virtual void setParent(const IGroupPtr& group) = 0;
};

} //namespace CDMACore
#endif //__CDMA_ICONTAINER_H__
