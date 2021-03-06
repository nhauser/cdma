package org.gumtree.data.soleil.array;

import org.gumtree.data.exception.BackupException;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IArrayIterator;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.interfaces.ISliceIterator;
import org.gumtree.data.math.IArrayMath;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.soleil.utils.NxsArrayMath;
import org.gumtree.data.soleil.utils.NxsArrayUtils;
import org.gumtree.data.utils.IArrayUtils;

public class NxsArrayMatrix implements NxsArrayInterface {
	private Object	       m_oData;        // It's an array of values
	private NxsIndexMatrix m_index;        // IIndex corresponding to this Array (dimension sizes defining the viewable part of the array)
	private IArray[]       m_arrays;       // Array of IArray
	private int[]          m_shapeMatrix;  // Shape of the matrix containing dataset
	private int[]          m_shapeItem;    // Shape of the dataitem: shape of matrix of dataset's + shape of the canonical dataset 

	public NxsArrayMatrix( IArray[] arrays ) {
		m_arrays = arrays;
		m_oData = null;
		initDimSize();
		
		// Define the same viewable part for all sub-IArray
		NxsIndex index = m_index.getIndexStorage();
		for( IArray array : arrays ) {
			array.setIndex(index);
		}
	}
	
	public NxsArrayMatrix( NxsArrayMatrix array ) {
		m_index       = (NxsIndexMatrix) array.m_index.clone();
		m_shapeMatrix = array.m_shapeMatrix.clone();
		m_shapeItem   = array.m_shapeItem.clone();
		m_oData       = array.m_oData;
		IIndex index  = m_index.getIndexStorage();
		m_arrays      = new IArray[array.m_arrays.length];
		for( int i = 0; i < array.m_arrays.length; i++ ) {
			m_arrays[i] = array.m_arrays[i].copy(false);
			m_arrays[i].setIndex(index);
			
		}
	}

    @Override
    public Object getObject(IIndex index) {
        return this.get(index);
    }

	@Override
	public IArray copy() {
		return copy(true);
	}
	
	@Override
	public IArray copy(boolean data) {
		NxsArrayMatrix result = new NxsArrayMatrix(this);
		
		if( data ) {
			result.m_oData = NxsArrayUtils.copyJavaArray(m_oData);
		}
		
		return result;
	}

    @Override
    public IArrayMath getArrayMath() {
        return new NxsArrayMath(this);
    }

    @Override
    public IArrayUtils getArrayUtils() {
        return new NxsArrayUtils(this);
    }
    
    @Override
    public boolean getBoolean(IIndex ima) {
        return (( Boolean ) get(ima)).booleanValue();
    }

    @Override
    public byte getByte(IIndex ima) {
        return (( Byte ) get(ima)).byteValue();
    }

    @Override
    public char getChar(IIndex ima) {
        return (( Character ) get(ima)).charValue();
    }

    @Override
    public double getDouble(IIndex ima) {
        return (( Double ) get(ima)).doubleValue();
    }

    @Override
    public float getFloat(IIndex ima) {
        return (( Float ) get(ima)).floatValue();
    }

    @Override
    public int getInt(IIndex ima) {
        return (( Integer ) get(ima)).intValue();
    }

    @Override
    public long getLong(IIndex ima) {
        return (( Long ) get(ima)).longValue();
    }

	@Override
	public Class<?> getElementType() {
		Class<?> result = null;
		if( m_arrays != null )
		{
			result = m_arrays[0].getElementType();
		}
		return result;
	}

	@Override
	public IIndex getIndex() {
		return m_index;
	}

	@Override
	public IArrayIterator getIterator() {
		return (IArrayIterator) new NxsArrayIterator(this);
	}

	@Override
	public int getRank() {
		return m_index.getRank();
	}

	@Override
	public IArrayIterator getRegionIterator(int[] reference, int[] range)
			throws InvalidRangeException {
	    IIndex index = new NxsIndexMatrix( m_shapeMatrix.length, m_index.getShape(), reference, range );
        return new NxsArrayIterator(this, index);
	}

	@Override
	public int[] getShape() {
		return m_index.getShape();
	}

	@Override
    public short getShort(IIndex ima) {
        return (( Short ) get(ima)).shortValue();
    }

	@Override
	public long getSize() {
		return m_index.getSize();
	}

	@Override
	public Object getStorage() {
		Object result = m_oData;
    	if( m_oData == null && m_arrays != null ) {
    		NxsIndex matrixIndex = (NxsIndex) m_index.getIndexMatrix().clone();
    		matrixIndex.set(new int[matrixIndex.getRank()]);

    		Long nbMatrixCells  = matrixIndex.getSize();
    		Long nbStorageCells = m_index.getIndexStorage().getSize();
    		int[] shape = { nbMatrixCells.intValue(), nbStorageCells.intValue() };
    		result = java.lang.reflect.Array.newInstance(getElementType(), shape);
    		
    		for( int i = 0; i < nbMatrixCells; i++ ) {
    			java.lang.reflect.Array.set(result, i, m_arrays[(int) matrixIndex.currentElement()].getStorage());
    			NxsArrayIterator.incrementIndex(matrixIndex);
    		}
    	}
    	
		return result;
	}

	@Override
	public void setBoolean(IIndex ima, boolean value) {
		set(ima, value);
	}

	@Override
	public void setByte(IIndex ima, byte value) {
        set(ima, value);
	}

	@Override
	public void setChar(IIndex ima, char value) {
        set(ima, value);
	}

	@Override
	public void setDouble(IIndex ima, double value) {
        set(ima, value);
	}

	@Override
	public void setFloat(IIndex ima, float value) {
        set(ima, value);
	}

	@Override
	public void setInt(IIndex ima, int value) {
        set(ima, value);
	}

	@Override
	public void setLong(IIndex ima, long value) {
	    set(ima, value);
	}

	@Override
	public void setObject(IIndex ima, Object value) {
        set(ima, value);
	}

	@Override
	public void setShort(IIndex ima, short value) {
        set(ima, value);
	}

	@Override
	public String shapeToString() {
        int[] shape = getShape();
        if (shape.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        sb.append('(');
        for (int i = 0; i < shape.length; i++) {
          int s = shape[i];
          if (i > 0) sb.append(",");
          sb.append(s);
        }
        sb.append(')');
        return sb.toString();
	}

    @Override
    public void setIndex(IIndex index) {
    	if( index instanceof NxsIndexMatrix ) {
    		m_index = (NxsIndexMatrix) index;
    	}
    	else {
    		// TODO !!!!!!! if any problem with setIndex : index.getShape() ---> index.getStride()
    		m_index = new NxsIndexMatrix(m_shapeMatrix.length, index.getShape(), index.getOrigin(), index.getShape() );
    		m_index.set(index.getCurrentCounter());
    	}
    }

	@Override
	public ISliceIterator getSliceIterator(int rank)
			throws ShapeNotMatchException, InvalidRangeException {
		return new NxsSliceIterator(this, rank);
	}

	@Override
	public void releaseStorage() throws BackupException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public long getRegisterId() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void lock() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void unlock() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean isDirty() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public IArray setDouble(double value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}

	@Override
	public String toString() {
		return m_index.toString();
	}
	
    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    /**
	 * InitDimSize
	 * Initialize member dimension sizes 'm_iDimS' according to defined member data 'm_oData'
	 */
	private void initDimSize()
	{
		// Check data existence
		if( m_arrays != null ) {
			// Set dimension rank
			int[] shape   = new int[ 1 + m_arrays[0].getRank() ];
			m_shapeMatrix = new int[ 1 ];
			m_shapeItem   = new int[ m_arrays[0].getRank() ];

			// Fill dimension size array
			m_shapeMatrix[0] = m_arrays.length;
			shape[0] = m_arrays.length;

			int i = 0;
			for( int size : m_arrays[0].getShape() ) {
				shape[i + 1] = size;
				m_shapeItem[i++] = size;
			}
			
			m_index  = new NxsIndexMatrix( m_shapeMatrix.length, shape, new int[shape.length], shape );
		}
	}
	
	/**
     * Get the object targeted by given index and return it (eventually using auto-boxing).
     * It's the central data access method that all other methods rely on.
     * 
     * @param index targeting a cell 
     * @return the content of cell designed by the index
     * @throws InvalidRangeException if one of the index is bigger than the corresponding dimension shape
     */
    private Object get(IIndex index) {
    	NxsIndexMatrix idx = null;
    	if( ! (index instanceof NxsIndexMatrix) ) {
    		idx = new NxsIndexMatrix(m_shapeMatrix.length, index);
    	}
    	else {
    		idx = (NxsIndexMatrix) index;
    	}

    	Object result = null;
    	NxsIndex itemIdx = idx.getIndexStorage();
    	long nodeIndex = idx.currentElementMatrix();
		
		IArray slab = m_arrays[(int) nodeIndex];
		
		if( slab != null ) {
			result = slab.getObject(itemIdx);
		}
		
		return result;
    }
    
    private void set(IIndex index, Object value) {
    	NxsIndexMatrix idx = null;
    	if( ! (index instanceof NxsIndexMatrix) ) {
    		idx = new NxsIndexMatrix(m_shapeMatrix.length, index);
    	}
    	else {
    		idx = (NxsIndexMatrix) index;
    	}
    	
    	NxsIndex itemIdx = idx.getIndexStorage();
    	long nodeIndex = idx.currentElementMatrix();
		IArray slab = m_arrays[(int) nodeIndex];
		if( slab != null ) {
			slab.setObject(itemIdx, value);
		}
    }

	@Override
	public void setDirty(boolean dirty) {
		// TODO Auto-generated method stub
		
	}
}
