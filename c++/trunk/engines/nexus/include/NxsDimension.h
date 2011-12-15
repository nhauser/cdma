//*****************************************************************************
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
//*****************************************************************************

#ifndef __CDMA_NXSDIMENSION_H__
#define __CDMA_NXSDIMENSION_H__

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDimension.h>

#include <internal/common.h>

namespace cdma
{

//=============================================================================
/// IDimension implementation
//=============================================================================
class NxsDimension : public IDimension
{
public:
  
  //@{ IDimension interface
	std::string getName();
	int getLength();
	bool isUnlimited();
	bool isVariableLength();
	bool isShared();
	Shape getCoordinateVariable();
	int hashCode();
	std::string toString();
	int compareTo(const IDimensionPtr& o);
	std::string writeCDL(bool strict);
	void setUnlimited(bool b);
	void setVariableLength(bool b);
	void setShared(bool b);
	void setLength(int n);
	void setName(const std::string& name);
	void setCoordinateVariable(const IArrayPtr& array) throw ( cdma::Exception );
  //@}

  //@{ IObject interface
  CDMAType::ModelType getModelType() const { return CDMAType::Dimension; };
  std::string getFactoryName() const { return NXS_FACTORY_NAME; };
  //@}
 };
}
#endif
