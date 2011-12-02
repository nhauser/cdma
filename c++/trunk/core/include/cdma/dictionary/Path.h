//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_PATH_H__
#define __CDMA_PATH_H__

#include <list>
#include <string>
#include "yat/memory/SharedPtr.h"

#include "cdma/IObject.h"
#include "cdma/dictionary/IPathMethod.h"
#include "cdma/dictionary/PathParameter.h"

namespace cdma
{

//==============================================================================
/// class Path
/// The IPath is an interface that describe how to reach a specific item in the
/// IDataset. The way that path are written is format plug-in dependent.
/// The more often it will be a <code>String</code> with a separator between nodes
/// that is relevant for the plug-in.
/// <p>
/// The string representation will only be interpreted by the plug-in. But it can be modified
/// using some parameters to make selective choice while browsing the nodes structure of the
/// IDataset.
/// <p>
/// In other cases (for the extended dictionary mechanism) it can also be (or be completed by)
/// a call on a plug-in method. It permits to describe from the dictionary how to have
/// access to the requested node.
/// A Group is a collection of DataItems. The Groups in a Dataset form a
/// hierarchical tree, like directories on a disk. A Group has a name and
/// optionally a set of Attributes.
//==============================================================================
class Path
{
private:
  std::string m_path;
  
public:
  // c-tor
  Path() {}
  
  //destructor
  ~Path() {};
  
  /// Set the value of the path in string
  ///
  void setValue(const std::string& path) { m_path = path; }

  /// Get the value of the path
  /// 
  /// @return string path value
  ///
  std::string getValue() { return m_path; }

  /// Clone the path so it keep unmodified when updated while it
  /// has an occurrence in a dictionary
  /// @return
  ///
  PathPtr clone();

  /// @return the string representation of the path
  ///
  std::string toString();

  /// Getter on methods that should be invoked to get data.
  /// The returned list is unmodifiable, key is the method and
  /// value is an array of Object arguments.
  ///  
  /// @return unmodifiable list of methods
  ///
  std::list<IPathMethod> getMethods();

  /// Set methods that should be invoked to get data.
  /// The list contains PathMethod having method and array of Object for
  /// arguments.
  /// 
  /// @param list that will be copied to keep it unmodifiable
  /// @return unmodifiable map of methods
  ///
  void setMethods(std::list<IPathMethod> methods);

  /// Will modify the path to make given parameters efficient.
  /// 
  /// @param parameters list of parameters to be inserted
  ///
  void applyParameters(std::list<PathParameterPtr> parameters);

  /// Will modify the path to remove all traces of parameters 
  /// that are not defined.
  ///
  void removeUnsetParameters();

  /// Will modify the path to unset all parameters that were defined
  ///
  void resetParameters();

  /// Analyze the path to reach the first undefined parameter.
  /// Return the path parameter to open the node. The parameter has 
  /// a wildcard for value (i.e: all matching nodes can be opened
  /// 
  /// @param param 
  ///           output path that will be updated with the appropriate node's
  ///           type and name until to reach the first path parameter
  /// 
  /// @return IPathParameter 
  /// 			 having the right type and name and a wildcard for value (empty)
  ///
  PathParameterPtr getFirstPathParameter(const std::string& output);
};

} //namespace CDMACore
#endif //__CDMA_PATH_H__
