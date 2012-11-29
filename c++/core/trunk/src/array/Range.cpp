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

//-----------------------------------------------------------------------------
// DEPENDENCIES
//-----------------------------------------------------------------------------
#include <cdma/array/impl/Range.h>
#include <cdma/exception/impl/ExceptionImpl.h>

namespace cdma
{
//---------------------------------------------------------------------------
// Range::Range
//---------------------------------------------------------------------------

Range::Range()
{
  m_last     = 0;
  m_first    = 0;
  m_stride   = 1;
  m_name     = "";
  m_length   = 1;
  m_reduced  = false;
}

//---------------------------------------------------------------------------
// Range::Range
//---------------------------------------------------------------------------
Range::Range(int length)
{
  set( length);
}

//---------------------------------------------------------------------------
// Range::Range
//---------------------------------------------------------------------------

Range::Range(std::string name, long first, long last, long stride, bool reduced)
{
  set(name, first, last, stride, reduced);
}

//---------------------------------------------------------------------------
// Range::Range
//---------------------------------------------------------------------------
Range::Range( const Range& range )
{
  set(range);
}

//---------------------------------------------------------------------------
// Range::set
//---------------------------------------------------------------------------
void Range::set(int length)
{
  //CDMA_FUNCTION_TRACE("Range::set( length)");
  m_name     = "";
  m_first    = 0;
  m_stride   = 1;
  m_last     = length - 1;
  m_length   = length;
  m_reduced  = false;
}

//---------------------------------------------------------------------------
// Range::set
//---------------------------------------------------------------------------
void Range::set(std::string name, long first, long last, long stride, bool reduced)
{
  //CDMA_FUNCTION_TRACE("Range::set(std::string name, long first, long last, long stride, bool reduced)");
  m_last     = last;
  m_first    = first;
  m_stride   = stride;
  m_name     = name;
  if( m_last < m_first )
  {
    m_length = ((m_last - m_first) / m_stride) - 1;
  }
  else
  {
    m_length = ((m_last - m_first) / m_stride) + 1;
  }
  
  m_reduced  = reduced;
}

//---------------------------------------------------------------------------
// Range::set
//---------------------------------------------------------------------------
void Range::set( const Range& range )
{
  //CDMA_FUNCTION_TRACE("Range::set(std::string name, long first, long last, long stride, bool reduced)");
  m_last     = range.m_last;
  m_first    = range.m_first;
  m_stride   = range.m_stride;
  m_name     = range.m_name;
  m_length   = range.m_length;
  m_reduced  = range.m_reduced;
}

//---------------------------------------------------------------------------
// Range::compose
//---------------------------------------------------------------------------
RangePtr Range::compose(const Range& range) throw ( Exception )
{
  Range* res = NULL;
  if ((length() == 0) || (range.length() == 0))
  {
    res = new Range();
  }
  else
  {
    long first = element(range.first());
    long strid = stride() * range.stride();
    long last  = element(range.last());
    res = new Range( m_name, first, last, strid);
  }
  return RangePtr(res);
}

//---------------------------------------------------------------------------
// Range::compact
//---------------------------------------------------------------------------
RangePtr Range::compact() throw ( Exception )
{
  long first, last, stride;
  std::string name;

  stride = 1;
  first  = m_first / m_stride;
  last   = m_last  / m_stride;
  name   = m_name;

  return  RangePtr(new Range(name, first, last, stride));
}

//---------------------------------------------------------------------------
// Range::shiftOrigin
//---------------------------------------------------------------------------
RangePtr Range::shiftOrigin(int origin) throw ( Exception )
{
  return RangePtr(new Range(m_name, m_first + origin, m_last + origin, m_stride,
          m_reduced ));
}

//---------------------------------------------------------------------------
// Range::intersect
//---------------------------------------------------------------------------
RangePtr Range::intersect(const Range& r) throw ( Exception )
{
  if ((length() == 0) || (r.length() == 0))
  {
    return  RangePtr ( new Range());
  }

  long last = this->last() > r.last() ? r.last() : this->last();
  long stride = this->stride() * r.stride();
  long useFirst;
  if (stride == 1)
  {
    useFirst = this->first() > r.first() ? this->first() : r.first();
  }
  else if (this->stride() == 1)
  { // then r has a stride
    if (r.first() >= first())
    {
      useFirst = r.first();
    }
    else
    {
      long incr = (first() - r.first()) / stride;
      useFirst = r.first() + incr * stride;
      if (useFirst < first())
        useFirst += stride;
    }
  }
  else if (r.stride() == 1)
  { // then this has a stride
    if (first() >= r.first())
    {
      useFirst = first();
    }
    else
    {
      long incr = (r.first() - first()) / stride;
      useFirst = first() + incr * stride;
      if (useFirst < r.first())
        useFirst += stride;
    }
  }
  else
  {
    THROW_INVALID_RANGE("Intersection when both ranges have a stride", "Range::intersect");
  }
  if (useFirst > last)
  {
    return  RangePtr(new Range());
  }
  else 
  {
    return  RangePtr(new Range(m_name, useFirst, last, stride));
  }
}

//---------------------------------------------------------------------------
// Range::intersects
//---------------------------------------------------------------------------
bool Range::intersects(const Range& r)
{
  if ((length() == 0) || (r.length() == 0))
  {
#ifdef CDMA_STD_SMART_PTR
    return  bool(RangePtr ( new Range()));
#else
    return RangePtr( new Range() );
#endif
  }

  long last = this->last() > r.last() ? r.last() : this->last();
  long stride = this->stride() * r.stride();
  long useFirst;
  if (stride == 1) 
  {
    useFirst = this->first() > r.first() ? this->first() : r.first();
  }
  else if (this->stride() == 1) // then r has a stride
  { 
    if (r.first() >= first()) 
    {
      useFirst = r.first();
    }
    else 
    {
      long incr = (first() - r.first()) / stride;
      useFirst = r.first() + incr * stride;
      if (useFirst < first())
        useFirst += stride;
    }
  }
  else if (r.stride() == 1) // then this has a stride
  { 
    if (first() >= r.first()) 
    {
      useFirst = first();
    }
    else 
    {
      long incr = (r.first() - first()) / stride;
      useFirst = first() + incr * stride;
      if (useFirst < r.first())
        useFirst += stride;
    }
  }
  else 
  {
    THROW_INVALID_RANGE("Intersection when both ranges have a stride", "Range::intersects");
  }
  return (useFirst <= last);
}

//---------------------------------------------------------------------------
// Range::unionRanges
//---------------------------------------------------------------------------
RangePtr Range::unionRanges(const Range& r) throw ( Exception )
{
  if( r.stride() != m_stride ) 
  {
    THROW_INVALID_RANGE("Stride must identical to make a Range union!", "Range::unionRanges");
  }

  if (this->length() == 0) 
  {
    return RangePtr(new Range(r));
  }

  if (r.length() == 0) 
  {
    return RangePtr(new Range(* this));
  }

  long first, last;
  std::string name = m_name;

  // Seek the smallest value
  first = m_first < r.first() ? m_first : r.first();
  last  = m_last > r.last() ? m_last : r.last();

  return RangePtr(new Range(name, first, last, m_stride));
}

//---------------------------------------------------------------------------
// Range::length
//---------------------------------------------------------------------------
int Range::length() const
{
  //CDMA_FUNCTION_TRACE("Range::length");
  return m_length;
}

//---------------------------------------------------------------------------
// Range::element
//---------------------------------------------------------------------------
int Range::element(int i) throw ( Exception )
{
  //CDMA_FUNCTION_TRACE("Range::element(int)");
  if ( i < 0 )
  {
    THROW_INVALID_RANGE("index must be >= 0", "Range::element");
  }
  if ( i > m_length )
  {
    THROW_INVALID_RANGE("index must be < length", "Range::element");
  }
  return (m_first + i * m_stride);
}

//---------------------------------------------------------------------------
// Range::index
//---------------------------------------------------------------------------
int Range::index(long& elem) throw ( Exception )
{
  //CDMA_FUNCTION_TRACE("Range::index(int)");
  if (elem < m_first) 
  {
    THROW_INVALID_RANGE("elem must be >= first", "Range::index");
  }
  int result = (int) ((elem - m_first) / m_stride);
  elem = (elem - m_first) % m_stride;
  if (result > m_last) 
  {
    THROW_INVALID_RANGE("elem must be <= last = n * stride", "Range::index");
  }
  return result;
}

//---------------------------------------------------------------------------
// Range::contains
//---------------------------------------------------------------------------
bool Range::contains(int i) const
{
  if( i < first() )
  {
    return false;
  }
  if( i > last()) 
  {
    return false;
  }
  if( m_stride == 1) 
  {
    return true;
  }
  return (i - m_first) % m_stride == 0;
}

//---------------------------------------------------------------------------
// Range::first
//---------------------------------------------------------------------------
int Range::first() const
{
  //CDMA_FUNCTION_TRACE("Range::first");
  return m_first;
}

//---------------------------------------------------------------------------
// Range::last
//---------------------------------------------------------------------------
int Range::last() const
{
//CDMA_FUNCTION_TRACE("Range::last");
  return m_last;
}

//---------------------------------------------------------------------------
// Range::stride
//---------------------------------------------------------------------------
int Range::stride() const
{
  //CDMA_FUNCTION_TRACE("Range::stride");
  return m_stride;
}

//---------------------------------------------------------------------------
// Range::getName
//---------------------------------------------------------------------------
std::string Range::getName() const
{
  return m_name;
}

//---------------------------------------------------------------------------
// Range::getFirstInterval
//---------------------------------------------------------------------------
int Range::getFirstInInterval(int start)
{
  if (start > last()) 
  {
    return -1;
  }
  if (start <= m_first) 
  {
    return m_first;
  }
  if (m_stride == 1) 
  {
    return start;
  }
  long offset = start - m_first;
  long incr = offset % m_stride;
  long result = start + incr;
  return ((result > last()) ? -1 : result);
}
}
