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


#include <cdma/exception/Exception.h>
#include <cdma/array/impl/Index.h>
#define TEMP_EXCEPTION(a,b) throw cdma::Exception("TARTAMPION", a, b)
namespace cdma
{
  //---------------------------------------------------------------------------
  // Index::Index
  //---------------------------------------------------------------------------
  Index::Index(cdma::IIndexPtr index)
  {
    CDMA_DBG("[BEGIN] Index::Index")
    m_rank    = index->getRank();
    m_vCurPos.resize(m_rank);
    m_pRanges.resize(m_rank);
    m_factory = index->getFactoryName();

    // Create new ranges
    std::vector<int> pos = index->getCurrentCounter();
    std::vector<int> ori = index->getOrigin();
    std::vector<int> sha = index->getShape();
    std::vector<int> str = index->getStride();
    for( int i = 0; i < m_rank; i++ )
    {
      m_vCurPos[i] = pos[i];
      m_pRanges[i].set(
        m_factory,
        index->getIndexName(i),
        ori[i],
        ori[i] + sha[i] * str[i],
        str[i]
      );
    }
    m_lastIndex = false;
    CDMA_DBG("[END] Index::Index")
  }

  Index::Index(const std::string factoryName, int rank, int shape[], int start[])
  {
    CDMA_DBG("[BEGIN] Index::Index(int rank, int shape[], int start[])")
    m_rank    = rank;
    m_vCurPos.resize(m_rank);
    m_pRanges.resize(m_rank);
    m_factory = factoryName;

    // Create new ranges
    long stride = 1;
    for( int i = m_rank - 1; i >= 0; i-- )
    {
      m_pRanges[i].set(
        m_factory,
        "",
        start[i] * stride,
        (start[i] + shape[i] - 1) * stride,
        stride
      );
      stride *= shape[i];
    }
    m_lastIndex = false;
    CDMA_DBG("[END] Index::Index")
  }

  //---------------------------------------------------------------------------
  // Index::getRank
  //---------------------------------------------------------------------------
  Index::~Index()
  {
    CDMA_DBG("[BEGIN] Index::~Index");
    CDMA_DBG("[END] Index::~Index");
  };

  //---------------------------------------------------------------------------
  // Index::getRank
  //---------------------------------------------------------------------------
  int Index::getRank()
  {
    return m_rank;
  }

  //---------------------------------------------------------------------------
  // Index::getShape
  //---------------------------------------------------------------------------
  std::vector<int> Index::getShape()
  {
    std::vector<int> shape(m_rank);
    for( int i = 0; i < m_rank; i++ )
    {
    shape[i] = m_pRanges[i].length();
    }
    return shape;
  }

  //---------------------------------------------------------------------------
  // Index::getOrigin
  //---------------------------------------------------------------------------
  std::vector<int> Index::getOrigin()
  {
    std::vector<int> origin(m_rank);
    for( int i = 0; i < m_rank; i++ )
    {
      origin[i] = (int) (m_pRanges[i].first() / m_pRanges[i].stride());
    }
    return origin;
  }

  //---------------------------------------------------------------------------
  // Index::getSize
  //---------------------------------------------------------------------------
  long Index::getSize()
  {
    long size = 1;
    if( m_rank == 0 )
    {
      size = 0;
    }
    else
    {
      for(int i = 0; i < m_rank; i++ )
      {
        size *= m_pRanges[i].length();
      }
    }
    return size;
  }

  //---------------------------------------------------------------------------
  // Index::getStride
  //---------------------------------------------------------------------------
  std::vector<int> Index::getStride()
  {
    std::vector<int> stride(m_rank);
    for( int i = 0; i < m_rank; i++ )
    {
      stride[i] = m_pRanges[i].stride();
    }
    return stride;
  }

  //---------------------------------------------------------------------------
  // Index::currentElement
  //---------------------------------------------------------------------------
  int Index::currentElement()
  {
    int value = 0;
    try {
      for( int i = 0; i < m_rank; i++ )
      {
        value += (m_pRanges[i]).element( m_vCurPos[i] );
      }
    } catch (cdma::Exception& e)
    {
      value = -1;
    }
    return value;
  }

  //---------------------------------------------------------------------------
  // Index::lastElement
  //---------------------------------------------------------------------------
  int Index::lastElement()
  {
    if( ! m_upToDate )
    {
      int last = 0;
      for( int i = 0; i < m_rank; i++ )
      {
      last += m_pRanges[i].last();
      }
      m_lastIndex = last;
      m_upToDate = true;
    }
    return m_lastIndex;
  }

  //---------------------------------------------------------------------------
  // Index::set
  //---------------------------------------------------------------------------
  void Index::set(std::vector<int> position)
  {
    if( position.size() != m_rank )
    {
      TEMP_EXCEPTION("Given position doesn't match to index's rank!", "Index::set");
    }

    for( int i = 0; i < m_rank; i++ )
    {
      m_vCurPos[i] = position[i];
    }
  }

  //---------------------------------------------------------------------------
  // Index::setDim
  //---------------------------------------------------------------------------
  void Index::setDim(int dim, int value)
  {
    if( dim >= m_rank || dim < 0 )
    {
      TEMP_EXCEPTION("Requested dimension doesn't exist!", "Index::setDim");
    }
    else if ( value < 0 || value >= m_pRanges[dim].length() )
    {

    }

    m_vCurPos[dim] = value;
  }

  //---------------------------------------------------------------------------
  // Index::setOrigin
  //---------------------------------------------------------------------------
  void Index::setOrigin(std::vector<int> origin)
  {/*
    if( origin.size() != m_rank )
    {
      TEMP_EXCEPTION("Origin must have same length as index's rank!", "Index::setOrigin");
    }
    int i = 0;
    while( i < origin.size() )
    {
      m_pRanges[i].set(
      Range range (
      m_factory,
      m_pRanges[i].getName(),
      (long) origin[i] * m_pRanges[i].stride(),
      (long) origin[i] * m_pRanges[i].stride() + (m_pRanges[i].length() - 1) * m_pRanges[i].stride() ,
      (long) m_pRanges[i].stride()
      );
      m_pRanges[i] = range;
      i++;
    }
    m_upToDate = false;*/
  }

  //---------------------------------------------------------------------------
  // Index::setShape
  //---------------------------------------------------------------------------
  void Index::setShape(std::vector<int> shape)
  {/*
    if( shape.size() != m_rank )
    {
      TEMP_EXCEPTION("Origin must have same length as index's rank!", "Index::setShape");
    }

    m_upToDate = false;
    int i = 0;
    while( i < shape.size() )
    {
      Range range (
      m_factory,
      m_pRanges[i].getName(),
      (long) m_pRanges[i].first(),
      (long) m_pRanges[i].first() + m_pRanges[i].stride() * shape[i],
      (long) m_pRanges[i].stride()
      );
      m_pRanges[i] = range;
      i++;
    }*/
  }

  //---------------------------------------------------------------------------
  // Index::setStride
  //---------------------------------------------------------------------------
  void Index::setStride(std::vector<int> stride)
  {/*
    if( stride.size() != m_rank )
    {
      TEMP_EXCEPTION("Origin must have same length as index's rank!", "Index::setStride");
    }

    m_upToDate = false;
    int i = 0;
    while( i < stride.size() )
    {
      Range range (
      m_factory,
      m_pRanges[i].getName(),
      (long) m_pRanges[i].first(),
      (long) m_pRanges[i].last(),
      (long) stride[i]
      );
      i++;
    }*/
  }

  //---------------------------------------------------------------------------
  // Index::getCurrentCounter
  //---------------------------------------------------------------------------
  std::vector<int> Index::getCurrentCounter()
  {
    std::vector<int> curPos (m_rank);

    for(int i = 0; i < m_rank; i++)
    {
      curPos[i] = m_vCurPos[i];
    }
    return curPos;
  }

  //---------------------------------------------------------------------------
  // Index::setIndexName
  //---------------------------------------------------------------------------
  void Index::setIndexName(int dim, const std::string& indexName)
  {
    if( dim >= m_rank || dim < 0 )
    {
      TEMP_EXCEPTION("Requested range is out of rank!", "Index::setIndexName");
    }
    Range range ( m_factory, indexName, m_pRanges[dim].first(), m_pRanges[dim].last(), m_pRanges[dim].stride() );
    m_pRanges[dim] = range;
  }

  //---------------------------------------------------------------------------
  // Index::getIndexName
  //---------------------------------------------------------------------------
  std::string Index::getIndexName(int dim)
  {
    if( dim >= m_rank || dim < 0 )
    {
      TEMP_EXCEPTION("Requested range is out of rank!", "Index::getIndexName");
    }
    return m_pRanges[dim].getName();
  }

  //---------------------------------------------------------------------------
  // Index::reduce
  //---------------------------------------------------------------------------
  cdma::IIndexPtr Index::reduce()
  {
    int newRank = 0;
    int dim = 0;
    std::vector<int> pos;
    std::vector<int> ori;
    std::vector<int> sha;
    std::vector<int> str;
    std::vector<std::string> names;
    for( int i = 0; i < m_rank; i++ )
    {
      if( m_pRanges[i].length() != 1 )
      {
        newRank++;
        pos.push_back(m_vCurPos[dim]);
        ori.push_back(m_pRanges[dim].first());
        sha.push_back(m_pRanges[dim].length());
        str.push_back(m_pRanges[dim].stride());
        names.push_back(m_pRanges[dim].getName());
        dim++;
      }
    }

    if( newRank != 0 )
    {
      m_rank = newRank;
      m_pRanges.resize(m_rank);
      for( int i = 0; i < m_rank; i++ )
      {
        m_pRanges[i].set( m_factory, names[i], ori[i] * str[i], sha[i] * str[i], str[i] );
      }
      m_vCurPos = pos;
      m_upToDate = false;
    }

    return cdma::IIndexPtr((cdma::IIndex*) this );
  }

  //---------------------------------------------------------------------------
  // Index::reduce
  //---------------------------------------------------------------------------
  cdma::IIndexPtr Index::reduce(int dim) throw ( cdma::Exception )
  {
    if( (dim < 0) || (dim >= m_rank) )
    {
      TEMP_EXCEPTION("Illegal reduce dim " + ('0' + dim), "Index::reduce" );
    }
    if( m_pRanges[dim].length() != 1 )
    {
      TEMP_EXCEPTION(std::string("Illegal reduce dim " + ('0' + dim)) + " : reduced dimension must be have length=1", "Index::reduce");
    }

    int newRank = m_rank - 1;
    std::vector<int> pos;
    std::vector<int> ori;
    std::vector<int> sha;
    std::vector<int> str;
    std::vector<std::string> names;
    for( int i = 0; i < m_rank; i++ )
    {
      if( i != dim )
      {
        newRank++;
        pos.push_back(m_vCurPos[i]);
        ori.push_back(m_pRanges[i].first());
        sha.push_back(m_pRanges[i].length());
        str.push_back(m_pRanges[i].stride());
        names.push_back(m_pRanges[i].getName());
        i++;
      }
    }
    if( newRank != 0 )
    {
      m_rank = newRank;
      m_pRanges.resize(m_rank);
      for( int i = 0; i < m_rank; i++ )
      {
        m_pRanges[i].set( m_factory, names[i], ori[i] * str[i], sha[i] * str[i], str[i] );
      }
      m_vCurPos = pos;
      m_upToDate = false;
    }

    return cdma::IIndexPtr((cdma::IIndex*) this );
  }

}
