package org.gumtree.data.soleil.array;

import java.util.ArrayList;
import java.util.List;

import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.interfaces.IRange;
import org.gumtree.data.soleil.NxsFactory;

public class NxsIndex implements IIndex {
	private int         m_rank;
    private int[]       m_iCurPos;          // Current position pointed by this index
    private NxsRange[]  m_ranges;           // Ranges that constitute the index global view in each dimension
    private boolean     m_upToDate = false; // Does the overall shape has changed
    private int         m_lastIndex;
    private long[]      m_projStride;
    private int[]       m_projShape;
    private int[]       m_projOrigin;

    /// Constructors
	public NxsIndex(fr.soleil.nexus4tango.DataItem ds) {
        this(ds.getSize(), new int[ds.getSize().length], ds.getSize());
    }

	public NxsIndex(int[] shape) {
        this(shape, new int[shape.length], shape);
	}
    
    public NxsIndex(NxsIndex index) {
    	m_rank    = index.m_rank;
        m_ranges  = new NxsRange[index.m_ranges.length];
        m_iCurPos = index.m_iCurPos.clone();
        m_projStride = index.m_projStride.clone();
        m_projShape  = index.m_projShape.clone();
        m_projOrigin = index.m_projOrigin.clone();
        for( int i = 0; i < index.m_ranges.length; i++ ) {
            m_ranges[i] = (NxsRange) index.m_ranges[i].clone();
        }
        m_lastIndex = index.m_lastIndex;
    }

    public NxsIndex(int[] shape, int[] start, int[] length) {
        long stride   = 1;
        m_rank       = shape.length;
        m_iCurPos    = new int[m_rank];
        m_ranges     = new NxsRange[m_rank];
        for( int i = m_rank - 1; i >= 0 ; i-- ) {
            try {
                m_ranges[i] = new NxsRange("", start[i] * stride, (start[i] + length[i] - 1) * stride, stride);
                stride *= shape[i];
            } catch( InvalidRangeException e ) {
                e.printStackTrace();
                m_ranges[i] = NxsRange.EMPTY;
            }
        }
        updateProjection();
    }
    
    public NxsIndex(List<IRange> ranges) {
    	m_rank       = ranges.size();
        m_iCurPos    = new int[m_rank];
        m_ranges     = new NxsRange[m_rank];
        int i = 0;
        for( IRange range : ranges ) {
        	m_ranges[i] = new NxsRange(range);
        	i++;
        }
        updateProjection();
    }
    
    // ---------------------------------------------------------
    /// Public methods
    // ---------------------------------------------------------
	@Override
	public long currentElement() {
        int value = 0;
        try {
            for( int i = 0; i < m_iCurPos.length; i++ ) {
                value += (m_ranges[i]).element( m_iCurPos[i] );
            }
        } catch (InvalidRangeException e) {
            value = -1;
        }        
        return value;
	}

	@Override
	public int[] getCurrentCounter() {
		int[] curPos = new int[m_rank];
        int i = 0;
        int j = 0;
        for(NxsRange range : m_ranges) {
        	if( ! range.reduced() ) {
        		curPos[i] = m_iCurPos[j];
        		i++;
        	}
        	j++;
        }
		return curPos;
	}

	@Override
	public String getIndexName(int dim) {
		return (m_ranges[dim]).getName();
	}

	@Override
	public int getRank() {
		return m_rank;
	}

	@Override
	public int[] getShape() {
        int[] shape = new int[m_rank];
        int i = 0;
        for(NxsRange range : m_ranges) {
        	if( ! range.reduced() ) {
        		shape[i] = range.length();
        		i++;
        	}
        }
		return shape;
	}

    @Override
    public long[] getStride() {
        long[] stride = new long[m_rank];
        int i = 0;
        for(NxsRange range : m_ranges) {
        	if( ! range.reduced() ) {
        		stride[i] = range.stride();
        		i++;
        	}
        }
        return stride;
    }
    
    @Override
    public int[] getOrigin() {
        int[] origin = new int[m_rank];
        int i = 0;
        for(NxsRange range : m_ranges) {
        	if( ! range.reduced() ) {
	        	origin[i] = (int) (range.first() / range.stride());
	            i++;
        	}
        }
        return origin;
    }
    
	@Override
	public long getSize() {
		if( m_ranges.length == 0 )
		{
			return 0;
		}

		long size = 1;
        for(NxsRange range : m_ranges) {
			size *= range.length();
		}

		return size;
	}
    
    @Override
    public void setOrigin(int[] origins) {
    	NxsRange range;
        int i = 0;
        int j = 0;
        while(  i < origins.length ) {
            range = m_ranges[j];
            if( ! range.reduced() ) {
            	range.last( origins[i] * range.stride() + (range.length() - 1) * range.stride() );
            	range.first( origins[i] * range.stride() );
        		i++;
        	}
        	j++;
        }
        updateProjection();
    }

	@Override
    public void setShape(int[] value) {
        NxsRange range;
        m_upToDate = false;
        int i = 0;
        int j = 0;
        while(  i < value.length ) {
            range = m_ranges[j];
            if( ! range.reduced() ) {
            	range.last( range.first() + (value[i] - 1) * range.stride() );
            	i++;
            }
            j++;
        }
        updateProjection();
    }
    
    @Override
    public void setStride(long[] stride) {
        NxsRange range;
        if( stride == null ) {
            return;
        }
        m_upToDate = false;
        int i = 0;
        int j = 0;
        while(  i < stride.length ) {
            range = m_ranges[j];
            if( ! range.reduced() ) {
            	range.stride( stride[i] );
            	i++;
            }
            j++;
        }
        updateProjection();
    }
    
	@Override
	public IIndex set(int[] index) {
	    if( index.length != m_rank )
            throw new IllegalArgumentException();
	    
        NxsRange range;
        int i = 0;
        int j = 0;
        while(  i < index.length ) {
            range = m_ranges[j];
            if( ! range.reduced() ) {
            	m_iCurPos[j] = index[i];
            	i++;
            }
            j++;
        }
	    
        return this;
	}

	@Override
	public IIndex set(int v0) {
		int[] iCurPos = new int[m_rank];
        iCurPos[0] = v0;
        this.set(iCurPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1) {
		int[] iCurPos = new int[m_rank];
        iCurPos[0] = v0;
        iCurPos[1] = v1;
        this.set(iCurPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2) {
		int[] iCurPos = new int[m_rank];
        iCurPos[0] = v0;
        iCurPos[1] = v1;
        iCurPos[2] = v2;
        this.set(iCurPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2, int v3) {
		int[] iCurPos = new int[m_rank];
        iCurPos[0] = v0;
        iCurPos[1] = v1;
        iCurPos[2] = v2;
        iCurPos[3] = v3;
        this.set(iCurPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2, int v3, int v4) {
		int[] iCurPos = new int[m_rank];
        iCurPos[0] = v0;
        iCurPos[1] = v1;
        iCurPos[2] = v2;
        iCurPos[3] = v3;
        iCurPos[4] = v4;
        this.set(iCurPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2, int v3, int v4, int v5) {
		int[] iCurPos = new int[m_rank];
        iCurPos[0] = v0;
        iCurPos[1] = v1;
        iCurPos[2] = v2;
        iCurPos[3] = v3;
        iCurPos[4] = v4;
        iCurPos[5] = v5;
        this.set(iCurPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2, int v3, int v4, int v5, int v6) {
		int[] iCurPos = new int[m_rank];
        iCurPos[0] = v0;
        iCurPos[1] = v1;
        iCurPos[2] = v2;
        iCurPos[3] = v3;
        iCurPos[4] = v4;
        iCurPos[5] = v5;
        iCurPos[6] = v6;
        this.set(iCurPos);
		return this;
	}

	@Override
	public IIndex set0(int v) {
        m_iCurPos[0] = v;
		return this;
	}

	@Override
	public IIndex set1(int v) {
        m_iCurPos[1] = v;
        return this;
	}

	@Override
	public IIndex set2(int v) {
        m_iCurPos[2] = v;
        return this;
	}

	@Override
	public IIndex set3(int v) {
        m_iCurPos[3] = v;
        return this;
	}

	@Override
	public IIndex set4(int v) {
        m_iCurPos[4] = v;
        return this;
	}

	@Override
	public IIndex set5(int v) {
        m_iCurPos[5] = v;
        return this;
	}

	@Override
	public IIndex set6(int v) {
        m_iCurPos[6] = v;
        return this;
	}

	@Override
	public void setDim(int dim, int value) {
        if( dim >= m_iCurPos.length || dim < 0 )
            throw new IllegalArgumentException();

		m_iCurPos[dim] = value;
	}

	@Override
	public void setIndexName(int dim, String indexName) {
        try {
            m_upToDate = false;
            int i = 0;
            int j = 0;
            NxsRange range = null;
            while( i <= dim ) {
           		range = m_ranges[j];
            	if( ! range.reduced() ) {
            		i++;
            	}
            	j++;
            }
            m_ranges[i - 1] = new NxsRange( indexName, range.first(), range.last(), range.stride() );
        } catch( InvalidRangeException e ) {
            e.printStackTrace();
        }
	}

    @Override
    public IIndex reduce() {
        for (int ii = 0; ii < m_iCurPos.length; ii++) {
        	// is there a dimension with length = 1 
        	if ( (m_ranges[ii]).length() == 1 && ! m_ranges[ii].reduced()) {
        		// remove it
        		this.reduce(0);
            
        		// ensure there is not any more to do
        		return this.reduce();
        	}
        }
        m_upToDate = false;
        return this;
    }


    /**
     * Create a new Index based on current one by
     * eliminating the specified dimension;
     *
     * @param dim: dimension to eliminate: must be of length one, else IllegalArgumentException
     * @return the new Index
     */
    @Override
    public IIndex reduce(int dim) {
    	// search the correct range
        int i = 0;
        NxsRange range = null;
        for(NxsRange rng : m_ranges) {
        	if( ! rng.reduced() ) {
        		if( i == dim ) {
        			range = rng;
        			break;
        		}
        		i++;
        	}
        }

        if( (dim < 0) || (dim >= m_rank) )
            throw new IllegalArgumentException("illegal reduce dim " + dim);
        if( range.length() != 1 )
            throw new IllegalArgumentException("illegal reduce dim " + dim + " : reduced dimension must be have length=1");
    
        // Reduce proper range
        range.reduced(true);
        m_rank--;
        updateProjection();
        return this;
    }

	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}

    @Override
    public String toString() {
        StringBuffer str = new StringBuffer();
        StringBuffer shp = new StringBuffer();
        str.append("Position: [");
        int i = 0;
        for( int pos : m_iCurPos ) {
            if( i++ != 0 ) {
                str.append(", ");
            }
            str.append(pos);
        }
        i = 0;
        str.append("]  <=> Index: " + currentElement() + "\nRanges:\n");
        shp.append("Shape [");
        for( NxsRange r : m_ranges ) {
        	if( !r.reduced() ) {
	        	if( i != 0 )
	        		shp.append(", ");
	        	shp.append(r.length());
	            str.append( "- nÂ°"+ i + " " + (NxsRange) r );
	            if( i < m_ranges.length ) {
	                str.append("\n");
	            }
        	}
        	i++;
        }
        shp.append("]\n");
        shp.append(str);
        return shp.toString();
    }
    
	@Override
	public String toStringDebug() {
        StringBuilder sbuff = new StringBuilder(100);
        sbuff.setLength(0);
        int rank = m_ranges.length;
        
        sbuff.append(" shape= ");
        for (int ii = 0; ii < rank; ii++) {
          sbuff.append( (m_ranges[ii]).length() );
          sbuff.append(" ");
        }

        sbuff.append(" stride= ");
        for (int ii = 0; ii < rank; ii++) {
          sbuff.append( (m_ranges[ii]).stride() );
          sbuff.append(" ");
        }

        long size   = 1;
        for (int ii = 0; ii < rank; ii++) {
            size  *= (m_ranges[ii]).length();
        }
        sbuff.append(" size= ").append(size);
        sbuff.append(" rank= ").append(rank);

        sbuff.append(" current= ");
        for (int ii = 0; ii < rank; ii++) {
          sbuff.append(m_iCurPos[ii]);
          sbuff.append(" ");
        }

        return sbuff.toString();	
    }
	
    @Override
	public long lastElement() {
        if( ! m_upToDate ) {
        	int last = 0;
            for( IRange range : m_ranges ) {
                last += range.last();
            }
            m_lastIndex = last;
            m_upToDate = true;
        }
        return m_lastIndex;
    }
    
    @Override
    public IIndex clone() {
    	NxsIndex index = new NxsIndex(this); 
    	return index;
    }
	
    // ---------------------------------------------------------
    /// Protected methods
    // ---------------------------------------------------------
    public List<IRange> getRangeList() {
        ArrayList<IRange> list = new ArrayList<IRange>();
        
        for( NxsRange range : m_ranges ) {
        	if( ! range.reduced() ) {
        		list.add(range);
        	}
        }
        return list;
    }
    
	protected int[] getCurrentPos() {
		return m_iCurPos;
	}
	
	protected int[] getProjectionShape() {
		return m_projShape;
	}
	
	protected int[] getProjectionOrigin() {
		return m_projOrigin;
	}
	
	public int currentProjectionElement() {
        int value = 0;
 
        for( int i = 0; i < m_iCurPos.length; i++ ) {
            value += m_iCurPos[i] * m_projStride[i];
        }
        return value;
	}
	
	private void updateProjection() {
		int realRank = m_ranges.length;
		m_projStride = new long[realRank];
		m_projShape  = new int[realRank];
        m_projOrigin = new int[realRank];
		long stride = 1;
		for( int i = realRank - 1; i >= 0; i-- ) {
			NxsRange range = m_ranges[i];
			m_projStride[i] = stride;
			m_projOrigin[i] = (int) (range.first() / range.stride());
			m_projShape[i]  = range.length();
			stride *= range.reduced() ? 1 : range.length();
		}
	}
}
