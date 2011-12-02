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

#include <cdma/array/impl/Range.h>

namespace cdma
{
  #define TEMP_EXCEPTION(a,b) throw Exception("TARTAMPION", a, b)
  Range::Range()
  {
    m_last    = 0;
    m_first   = 0;
    m_stride  = 1;
    m_name    = "";
    m_reduced = false;
  }

  Range::Range(const std::string& factory, int length)
  {
    set( factory, length);
  }

  Range::Range(const std::string& factory, std::string name, long first, long last, long stride, bool reduced)
  {
    set(factory, name, first, last, stride, reduced);
  }

  Range::Range( const IRange& range )
  {
    set(range);
  }

  void Range::set(const std::string& factory, int length)
  {
    m_name    = "";
    m_first   = 0;
    m_stride  = 1;
    m_last    = length - 1;
    m_reduced = false;
    m_factory = factory;
  }

  void Range::set(const std::string& factory, std::string name, long first, long last, long stride, bool reduced)
  {
    m_last    = last;
    m_first   = first;
    m_stride  = stride;
    m_name    = name;
    m_reduced = reduced;
    m_factory = factory;
  }

  void Range::set( const IRange& range )
  {
    m_last    = range.last();
    m_first   = range.first();
    m_stride  = range.stride();
    m_name    = range.getName();
    m_reduced = false;
    m_factory = range.getFactoryName();
  }

  IRangePtr Range::compose(const IRange& range) throw ( Exception )
  {
    IRange* res = NULL;
    if ((length() == 0) || (range.length() == 0))
    {
      res = new Range();
    }
    else
    {
      long first = element(range.first());
      long strid = stride() * range.stride();
      long last  = element(range.last());
      res = new Range(range.getFactoryName(), m_name, first, last, strid);
    }
    return res;
  }

  IRangePtr Range::compact() throw ( Exception )
  {
    long first, last, stride;
    std::string name;

    stride = 1;
    first  = m_first / m_stride;
    last   = m_last  / m_stride;
    name   = m_name;

    return  IRangePtr ( (IRange*) new Range(m_factory, name, first, last, stride) );
  }

  IRangePtr Range::shiftOrigin(int origin) throw ( Exception )
  {
    return  IRangePtr ( (IRange*) new Range(m_factory,  m_name, m_first + origin, m_last + origin, m_stride, m_reduced ) );
  }

  IRangePtr Range::intersect(const IRange& r) throw ( Exception )
  {
    if ((length() == 0) || (r.length() == 0))
    {
      return  IRangePtr ( (IRange*) new Range());
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
      TEMP_EXCEPTION("Intersection when both ranges have a stride", "Range::intersect");
    }
    if (useFirst > last)
    {
      return  IRangePtr ( (IRange*) new Range());
    }
    else {
      return  IRangePtr ( (IRange*) new Range(m_factory, m_name, useFirst, last, stride) );
    }
  }

  bool Range::intersects(const IRange& r)
  {
    if ((length() == 0) || (r.length() == 0))
    {
      return  IRangePtr ( (IRange*) new Range());
    }

    long last = this->last() > r.last() ? r.last() : this->last();
    long stride = this->stride() * r.stride();
    long useFirst;
    if (stride == 1) {
      useFirst = this->first() > r.first() ? this->first() : r.first();
    }
    else if (this->stride() == 1) { // then r has a stride
      if (r.first() >= first()) {
        useFirst = r.first();
      }
      else {
        long incr = (first() - r.first()) / stride;
        useFirst = r.first() + incr * stride;
        if (useFirst < first())
          useFirst += stride;
      }
    }
    else if (r.stride() == 1) { // then this has a stride
      if (first() >= r.first()) {
        useFirst = first();
      }
      else {
        long incr = (r.first() - first()) / stride;
        useFirst = first() + incr * stride;
        if (useFirst < r.first())
          useFirst += stride;
      }
    }
    else {
      TEMP_EXCEPTION("Intersection when both ranges have a stride", "Range::intersects");
    }
    return (useFirst <= last);
  }

  IRangePtr Range::unionRanges(const IRange& r) throw ( Exception )
  {
    if( r.stride() != m_stride ) {
      TEMP_EXCEPTION("Stride must identical to make a IRange union!", "Range::unionRanges");
    }

    if (this->length() == 0) {
      return IRangePtr ( (IRange*) new Range(r) );
    }

    if (r.length() == 0) {
      return IRangePtr ( (IRange*) new Range(* this) );
    }

    long first, last;
    std::string name = m_name;

    // Seek the smallest value
    first = m_first < r.first() ? m_first : r.first();
    last  = m_last > r.last() ? m_last : r.last();

    return IRangePtr ( (IRange*) new Range(m_factory, name, first, last, m_stride) );
  }

  int Range::length() const
  {
    return (int) ((m_last - m_first) / m_stride) + 1;
  }

  int Range::element(int i) throw ( Exception )
  {
    if (i < 0) {
      TEMP_EXCEPTION("index must be >= 0", "Range::element");
    }
    if (i > m_last) {
      TEMP_EXCEPTION("index must be < length", "Range::element");
    }

    return (int) (m_first + i * m_stride);
  }

  int Range::index(int elem) throw ( Exception )
  {
    if (elem < m_first) {
      TEMP_EXCEPTION("elem must be >= first", "Range::element");
    }
    long result = (elem - m_first) / m_stride;
    if (result > m_last) {
      TEMP_EXCEPTION("elem must be <= last = n * stride", "Range::element");
    }
    return (int) result;
  }

  bool Range::contains(int i) const
  {
    if( i < first() ) {
      return false;
    }
    if( i > last()) {
      return false;
    }
    if( m_stride == 1) {
      return true;
    }
    return (i - m_first) % m_stride == 0;
  }

  int Range::first() const
  {
    return m_first;
  }

  int Range::last() const
  {
    return m_last;
  }

  int Range::stride() const
  {
    return m_stride;
  }

  std::string Range::getName() const
  {
    return m_name;
  }

  int Range::getFirstInInterval(int start)
  {
    if (start > last()) {
      return -1;
    }
    if (start <= m_first) {
      return (int) m_first;
    }
    if (m_stride == 1) {
      return start;
    }
    long offset = start - m_first;
    long incr = offset % m_stride;
    long result = start + incr;
    return (int) ((result > last()) ? -1 : result);
  }
}