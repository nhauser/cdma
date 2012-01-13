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
#ifndef __CDMA_IDATA_H__
#define __CDMA_IDATA_H__

#include <yat/any/Any.h>
#include <cdma/IObject.h>
#include <vector>

namespace cdma
{

class IArrayStorage : public IObject
{
public:
//  virtual ~Data() { std::cout<<"Data::~Data"<<std::endl; };

  virtual void* getStorage() = 0;
//  virtual yat::Any& get( long index ) = 0;
  virtual yat::Any& get( const cdma::ViewPtr view, std::vector<int> position ) = 0;
  virtual void set(const cdma::ViewPtr& view, std::vector<int> position, const yat::Any& value) = 0;
  virtual const std::type_info& getType() = 0;
  virtual bool dirty() = 0;
  virtual void setDirty(bool dirty) = 0;
};

}

#endif
