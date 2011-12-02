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
#ifndef __CDMA_RANGE_H__
#define __CDMA_RANGE_H__

#include <cdma/exception/Exception.h>
#include <cdma/array/IRange.h>



namespace cdma
{
//==============================================================================
/// Range default implementation
//==============================================================================
class Range : public cdma::IRange
{
 private:
  int         m_last;     // number of elements
  int         m_first;    // first value in range
  int         m_stride;   // stride, must be >= 1
  bool        m_reduced;  // was this ranged reduced or not
  std::string m_name;     // optional name
  std::string m_factory;  // Name of the factory
 
public:
	/// Constructors
	Range();
  Range( const std::string& factory,int length );
  Range( const std::string& factory, std::string name, long first, long last, long stride, bool reduced = false );
  Range( const cdma::IRange& range );
  
	~Range() { };

  //@{ internal commons method
  void set( const std::string& factory,int length );
  void set( const std::string& factory, std::string name, long first, long last, long stride, bool reduced = false );
  void set( const cdma::IRange& range );
  //@} internal commons method
  
  //@{ IRange interface
  cdma::IRangePtr     compose(const cdma::IRange& r)     throw ( cdma::Exception );
  cdma::IRangePtr     compact()                          throw ( cdma::Exception );
  cdma::IRangePtr     shiftOrigin(int origin)            throw ( cdma::Exception );
  cdma::IRangePtr     intersect(const cdma::IRange& r)   throw ( cdma::Exception );
  bool                intersects(const cdma::IRange& r);
  cdma::IRangePtr     unionRanges(const cdma::IRange& r) throw ( cdma::Exception );
  int                 length() const ;
  int                 element(int i)                     throw ( cdma::Exception );
  int                 index(int elem)                    throw ( cdma::Exception );
  bool                contains(int i) const;
  int                 first() const;
	int                 last() const;
	int                 stride() const;
	std::string         getName() const;
  int                 getFirstInInterval(int start);
  //@} IRange interface
  
  //@{ IObject interface
	std::string         getFactoryName() const  { return m_factory; };
  CDMAType::ModelType getModelType() const    { return cdma::CDMAType::Other; };
  //@} IObject interface
};
}
#endif
