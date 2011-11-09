package org.gumtree.data.soleil;

import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.interfaces.IRange;

public class NxsRange implements IRange {
    public static final NxsRange EMPTY = new NxsRange();
    public static final NxsRange VLEN  = new NxsRange(-1);

    /// Members
    private long      m_last;     // number of elements
    private long      m_first;    // first value in range
    private long      m_stride;   // stride, must be >= 1
    private String    m_name;     // optional name
    private boolean   m_reduced;  // was this ranged reduced or not

    /// Constructors
    /**
     * Used for EMPTY
     */
    private NxsRange() {
      this.m_last    = 0;
      this.m_first   = 0;
      this.m_stride  = 1;
      this.m_name    = null;
      this.m_reduced = false;
    }

    /**
     * Create a range starting at zero, with an unit stride of length "length".
     * @param length number of elements in the NxsRange
     */
    public NxsRange(int length) {
      this.m_name    = null;
      this.m_first   = 0;
      this.m_stride  = 1;
      this.m_last    = length - 1;
      this.m_reduced = false;
    }
    
    /**
     * Create a range with a specified stride.
     *
     * @param name   name of the range 
     * @param first  first index in range
     * @param last   last index in range, inclusive
     * @param stride stride between consecutive elements, must be > 0
     * @throws InvalidRangeException elements must be nonnegative: 0 <= first <= last, stride > 0
     */
    public NxsRange(String name, long first, long last, long stride) throws InvalidRangeException {
        this.m_last    = last;
        this.m_first   = first;
        this.m_stride  = stride;
        this.m_name    = name;
        this.m_reduced = false;
    }
    
    public NxsRange(String name, long first, long last, long stride, boolean reduced) throws InvalidRangeException {
    	this(name, first, last, stride);
    	m_reduced = reduced;
    }
    
    public NxsRange( IRange range ) {
        this.m_last    = range.last();
        this.m_first   = range.first();
        this.m_stride  = range.stride();
        this.m_name    = range.getName();
        this.m_reduced = false;
    }
    
    /// Accessors
    @Override
    public String getName() {
        return m_name;
    }

    @Override
    public long first() {
        return m_first;
    }
    
    @Override
    public long last() {
        return m_last;
    }

    @Override
    public int length() {
        return (int) ((m_last - m_first) / m_stride) + 1;
    }

    @Override
    public long stride() {
        return m_stride;
    }
    
    protected void stride(long value) {
    	m_stride = value;
    }
    
    protected void last(long value) {
    	m_last = value;
    }
    
    protected void first(long value) {
    	m_first = value;
    }
    
    /// Methods
    @Override
    public IRange clone() {
        NxsRange range = NxsRange.EMPTY;
        try {
            range = new NxsRange(m_name, m_first, m_last, m_stride);
            range.m_reduced = m_reduced;
        } catch( InvalidRangeException e ) {
            e.printStackTrace();
        }
        return range;
    }
    
	@Override
	public IRange compact() throws InvalidRangeException {
		long first, last, stride;
        String name;

        stride = 1;
        first  = m_first / m_stride;
        last   = m_last / m_stride;
        name   = m_name; 
        
        return new NxsRange(name, first, last, stride);
	}

	@Override
	public IRange compose(IRange r) throws InvalidRangeException {
        if ((length() == 0) || (r.length() == 0)) {
            return EMPTY;
        }
        if (this == VLEN || r == VLEN) {
            return VLEN;
        }

        long first  = element(r.first());
        long stride = stride() * r.stride();
        long last   = element(r.last());
        return new NxsRange(m_name, first, last, stride);
	}

	@Override
	public boolean contains(int i) {
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

	@Override
	public int element(long i) throws InvalidRangeException {
	    if (i < 0) {
            throw new InvalidRangeException("i must be >= 0");
        }
	    if (i > m_last) {
	        throw new InvalidRangeException("i must be < length");
	    }

        return (int) (m_first + i * m_stride);
	}

	@Override
	public int getFirstInInterval(int start) {
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

	@Override
	public long index(int elem) throws InvalidRangeException {
        if (elem < m_first) {
            throw new InvalidRangeException("elem must be >= first");
        }
        long result = (elem - m_first) / m_stride;
        if (result > m_last) {
            throw new InvalidRangeException("elem must be <= last = n * stride");
        }
        return (int) result;
	}

	@Override
	public IRange intersect(IRange r) throws InvalidRangeException {
        if ((length() == 0) || (r.length() == 0)) {
            return EMPTY;
        }
        if (this == VLEN || r == VLEN) {
            return VLEN;
        }

        long last = Math.min(this.last(), r.last());
        long stride = stride() * r.stride();

        long useFirst;
        if (stride == 1) {
            useFirst = Math.max(this.first(), r.first());
        }
        else if (stride() == 1) { // then r has a stride
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
            throw new UnsupportedOperationException("Intersection when both ranges have a stride");
        }
        if (useFirst > last) {
            return EMPTY;
        }
        return new NxsRange(m_name, useFirst, last, stride);
	}

	@Override
	public boolean intersects(IRange r) {
        if ((length() == 0) || (r.length() == 0)) {
            return false;
        }
        if (this == VLEN || r == VLEN) {
            return true;
        }

        long last = Math.min(this.last(), r.last());
        long stride = stride() * r.stride();

        long useFirst;
        if (stride == 1) {
            useFirst = Math.max(this.first(), r.first());
        }
        else if (stride() == 1) { // then r has a stride
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
            throw new UnsupportedOperationException("Intersection when both ranges have a stride");
        }
        return (useFirst <= last);
	}
    
	@Override
	public IRange shiftOrigin(int origin) throws InvalidRangeException {
        return new NxsRange( m_name, m_first + origin, m_last + origin, m_stride, m_reduced );
	}

	@Override
	public IRange union(IRange r) throws InvalidRangeException {
	    if( r.stride() != m_stride ) {
	        throw new InvalidRangeException("Stride must identical to make a IRange union!");
        }
        
        if (length() == 0) {
            return r;
        }
        if (this == VLEN || r == VLEN) {
            return VLEN;
        }
        if (r.length() == 0) {
            return this;
        }
        
        long first, last;
        String name = m_name;
        
        // Seek the smallest value
        first = Math.min( m_first, r.first() );
        last  = Math.max( m_last , r.last()  );
        
        return new NxsRange(name, first, last, m_stride);
	}

    @Override
    public String toString() {
        StringBuffer str = new StringBuffer();
        str.append("name: '" + getName());
        str.append("', first: " + first());
        str.append(", last: " + last());
        str.append(", stride: " + stride());
        str.append(", length: " + length());
        str.append(", reduce: " + m_reduced);
        return str.toString();
    }
    
	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}
	
	protected boolean reduced() {
		return m_reduced;
	}
	
	protected void reduced(boolean reduce) {
		m_reduced = reduce;
	}
}
