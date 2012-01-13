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

#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>

namespace cdma
{
//==============================================================================
/// Range default implementation
//==============================================================================
class Range
{
 private:
  int         m_last;     // offset of last element
  int         m_first;    // offset of first element
  int         m_stride;   // stride, must be >= 1
  bool        m_reduced;  // was this ranged reduced or not
  std::string m_name;     // optional name
 
public:
	/// Constructors
	Range();
  Range( int length );
  Range( std::string name, long first, long last, long stride, bool reduced = false );
  Range( const cdma::Range& range );
  
	~Range() { };

  //@{ internal commons method
  void set( int length );
  void set( std::string name, long first, long last, long stride, bool reduced = false );
  void set( const cdma::Range& range );
  //@} internal commons method
  
  //@{ IRange interface
  RangePtr    compose(const cdma::Range& r)     throw ( cdma::Exception );
  RangePtr    compact()                         throw ( cdma::Exception );
  RangePtr    shiftOrigin(int origin)           throw ( cdma::Exception );
  RangePtr    intersect(const cdma::Range& r)   throw ( cdma::Exception );
  bool        intersects(const cdma::Range& r);
  RangePtr    unionRanges(const cdma::Range& r) throw ( cdma::Exception );
  int         length() const ;
  int         element(int i)                    throw ( cdma::Exception );
  int         index(int elem)                   throw ( cdma::Exception );
  bool        contains(int i) const;
  int         first() const;
	int         last() const;
	int         stride() const;
	std::string getName() const;
  int         getFirstInInterval(int start);
  //@} IRange interface
  
  void setReduce(bool reduce) { m_reduced = reduce; };
  bool reduce() const      { return m_reduced; };
};
}
#endif
