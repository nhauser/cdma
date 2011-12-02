//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IDIMENSION_H__
#define __CDMA_IDIMENSION_H__

#include <yat/utils/String.h>
#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>

#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>

namespace cdma
{

//==============================================================================
/// Shape
/// Dimensions of a data item
//==============================================================================
typedef std::vector<int> Shape;
  
//==============================================================================
/// IDimension
/// A Dimension is used to define the array shape of a DataItem. It may be shared
/// among DataItems, which provides a simple yet powerful way of associating
/// DataItems.
//==============================================================================
class IDimension : public IObject
{
  public:
    virtual ~IDimension() {};
    
    /// Returns the name of this Dimension; may be null. A Dimension with a null
    /// name is called "anonymous" and must be private. Dimension names are
    /// unique within a Group.
    ///
    /// @return string object
    ///
    virtual std::string getName() = 0;
    
    /// Get the length of the Dimension.
    ///
    /// @return integer value
    ///
    virtual int getLength() = 0;
    
    /// If unlimited, then the length can increase; otherwise it is immutable.
    ///
    /// @return true or false
    ///
    virtual bool isUnlimited() = 0;
    
    /// If variable length, then the length is unknown until the data is read.
    ///
    /// @return true or false
    ///
    virtual bool isVariableLength() = 0;
    
    /// If this Dimension is shared, or is private to a Variable. All Dimensions
    /// in NetcdfFile.getDimensions() or Group.getDimensions() are shared.
    /// Dimensions in the Variable.getDimensions() may be shared or private.
    ///
    /// @return true or false
    ///
    virtual bool isShared() = 0;
    
    /// Get the coordinate variables or coordinate variable aliases if the
    /// dimension has any, else return an empty list. A coordinate variable has
    /// this as its single dimension, and names this Dimensions's the
    /// coordinates. A coordinate variable alias is the same as a coordinate
    /// variable, but its name must match the dimension name. If numeric,
    /// coordinate axis must be strictly monotonically increasing or decreasing.
    ///
    /// @return IArray containing coordinates
    ///
    virtual Shape getCoordinateVariable() = 0;
    
    /// Instances which have same contents are equal.
    ///
    /// @param oo
    ///            Object
    /// @return true or false
    ///
    virtual bool equals(const IDimensionPtr& other) = 0;
    
    /// Override Object.hashCode() to implement equals.
    ///
    /// @return integer value
    ///
    virtual int hashCode() = 0;
    
    /// string representation.
    ///
    /// @return string object
    ///
    virtual std::string toString() = 0;
    
    /// Dimensions with the same name are equal.
    ///
    /// @param o
    ///            compare to this Dimension
    /// @return 0, 1, or -1
    ///
    virtual int compareTo(const IDimension& o) = 0;
    
    /// string representation.
    ///
    /// @param strict
    ///            bool type
    /// @return string object
    ///
    virtual std::string writeCDL(bool strict) = 0;
    
    /// Set whether this is unlimited, meaning length can increase.
    ///
    /// @param b
    ///            bool type
    ///
    virtual void setUnlimited(bool b) = 0;
    
    /// Set whether the length is variable.
    ///
    /// @param b
    ///            bool type
    ///
    virtual void setVariableLength(bool b) = 0;
    
    /// Set whether this is shared.
    ///
    /// @param b
    ///            bool type
    ///
    virtual void setShared(bool b) = 0;
    
    /// Set the Dimension length.
    ///
    /// @param n
    ///            integer value
    ///
    virtual void setLength(int n) = 0;
    
    /// Rename the dimension.
    ///
    /// @param name
    ///            string object
    ///
    virtual void setName(const std::string& name) = 0;
    
    /// Set coordinates values for this dimension.
    ///
    /// @param array
    ///            with new coordinates
    ///
    //## virtual void setCoordinateVariable(IArray& array) throw ( ShapeNotMatchException ) = 0;
};

typedef yat::SharedPtr<IDimension, yat::Mutex> IDimensionPtr;

} //namespace cdma

#endif //__CDMA_IDIMENSION_H__
