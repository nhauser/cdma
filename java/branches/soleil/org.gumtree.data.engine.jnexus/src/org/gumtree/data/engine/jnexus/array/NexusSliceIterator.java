package org.gumtree.data.engine.jnexus.array;

import org.gumtree.data.engine.jnexus.NexusFactory;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.interfaces.ISliceIterator;

public class NexusSliceIterator implements ISliceIterator {

    /// Members
    private NexusArrayIterator  m_iterator;      // iterator of the whole_array
    private IArray              m_array;         // array of the original shape containing all slices
    private int[]               m_dimension;     // shape of the slice
    
    /// Constructor
    /**
     * Create a new NxsSliceIterator that permits to iterate over
     * slices that are the last dim dimensions of this array.
     * 
     * @param array source of the slice
     * @param dim returned dimensions
     */
    public NexusSliceIterator(final IArray array, final int dim) throws InvalidRangeException {
        // If ranks are equal, make sure at least one iteration is performed.
        // We cannot use 'reshape' (which would be ideal) as that creates a
        // new copy of the array storage and so is unusable
        m_array = array;
        int[] shape     = m_array.getShape();
        int[] rangeList = shape.clone();
        int[] origin    = m_array.getIndex().getOrigin().clone();
        long[] stride   = m_array.getIndex().getStride().clone();
        int   rank      = m_array.getRank();
        
        // shape to make a 2D array from multiple-dim array
        m_dimension = new int[dim];
        System.arraycopy(shape, shape.length - dim, m_dimension, 0, dim);
        for( int i = 0; i < dim; i++ ) {
            rangeList[rank - i - 1] = 1;
        }
        // Create an iterator over the higher dimensions. We leave in the
        // final dimensions so that we can use the getCurrentCounter method
        // to create an origin.
        
        IIndex index;
		try {
			index = m_array.getIndex().clone();
			
	        // As we get a reference on array's IIndex we directly modify it
	        index.setOrigin(origin);
	        index.setStride(stride);
	        index.setShape(rangeList);
	        
	        m_iterator = new NexusArrayIterator(m_array, index, false);
			
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
    }
    
    /// Public methods
	@Override
	public IArray getArrayCurrent() throws InvalidRangeException {
		return createSlice();
	}

    @Override
    public IArray getArrayNext() throws InvalidRangeException {
    	next();
    	IArray result = createSlice();
        return result;
    }

	@Override
	public int[] getSliceShape() throws InvalidRangeException {
		return m_dimension;
	}
	
	public int[] getSlicePosition() {
		return m_iterator.getCounter();
	}

	@Override
	public boolean hasNext() {
		return m_iterator.hasNext();
	}

	@Override
	public void next() {
		m_iterator.incrementIndex();
	}

	@Override
	public String getFactoryName() {
		return NexusFactory.NAME;
	}
	
    private IArray createSlice() throws InvalidRangeException {
    	int i = 0;
    	int[] iShape  = m_array.getShape();
    	int[] iOrigin = m_array.getIndex().getOrigin();
    	int[] iCurPos = m_iterator.getCounter();
    	for( int pos : iCurPos ) {
    		if( pos >= iShape[i] ) {
    			return null;
    		}
    		iOrigin[i] += pos;
    		i++;
    	}
    	
    	java.util.Arrays.fill(iShape, 1);
    	System.arraycopy(m_dimension, 0, iShape, iShape.length - m_dimension.length, m_dimension.length); 

        IArray slice = m_array.copy(false);
    	
        IIndex index = slice.getIndex();
        index.setShape(iShape);
        index.setOrigin(iOrigin);
        index.reduce();
        return slice;
    }

}
