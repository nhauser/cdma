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

import fr.soleil.nexus4tango.DataItem;


public class NxsArray implements NxsArrayInterface {
    private IIndex    m_index;        // IIndex corresponding to this Array (dimension sizes defining the viewable part of the array)
	private Object	  m_oData;        // It's an array of values
	private boolean	  m_bRawArray;    // True if the stored array has a rank of 1 (independently of its shape)
    private boolean   m_isDirty;      // Is the array synchronized with the handled file
    private DataItem  m_n4tdataitem;  // Array of datasets that are used to store the storage backing
    private int[]     m_shape;        // Shape of the array (dimension sizes of the storage backing) 
    
	// Constructors
	public NxsArray(Object oArray, int[] iShape) {
        m_index = new NxsIndex(iShape);
		m_oData	= oArray;
		m_shape	= iShape;
        if( iShape.length == 1 ) {
            m_bRawArray	= true;
        }
        else {
            m_bRawArray = (
                            oArray.getClass().isArray() && 
                            !(java.lang.reflect.Array.get(oArray, 0).getClass().isArray()) 
                          );
        }
        m_n4tdataitem = null;
	}

    public NxsArray(NxsArray array) {
        try {
			m_index = (NxsIndex) array.m_index.clone();
		} catch (CloneNotSupportedException e) {
			m_index = null;
		}
        m_oData = array.m_oData;
        m_shape = new int[array.m_shape.length];
        int i   = 0;
        for( int size : array.m_shape ) {
            m_shape[i++]  = size;
        }
        m_bRawArray  = array.m_bRawArray;
        m_isDirty    = array.m_isDirty;
        try {
			m_n4tdataitem = array.m_n4tdataitem.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
    }
    
    public NxsArray(DataItem ds) {
        m_index      = new NxsIndex(ds);
		m_oData      = null;
		m_shape      = ds.getSize();
		m_bRawArray  = ds.isSingleRawArray();
        m_isDirty    = false;
        m_n4tdataitem = ds;
	}
    
	// ---------------------------------------------------------
	/// public methods
	// ---------------------------------------------------------
    @Override
    public IArrayMath getArrayMath() {
        return new NxsArrayMath(this);
    }

    @Override
    public IArrayUtils getArrayUtils() {
        return new NxsArrayUtils(this);
    }
    
	/// Specific method to match NetCDF plug-in behavior
	@Override
	public String toString()
	{
		Object oData = getData();
		if( oData instanceof String )
			return (String) oData;
	    StringBuilder sbuff = new StringBuilder();
	    IArrayIterator ii = getIterator();
	    int i = 1;
        Object data = null;
	    while (ii.hasNext())
	    {
            data = ii.next();
            sbuff.append(data);
            sbuff.append(" ");
			i++;
	    }
	    return sbuff.toString().substring(0, sbuff.length() < 10000 ? sbuff.length() : 10000);
	}

	/// IArray underlying data access
	@Override
	public Object getStorage() {
		return getData();
	}

	@Override
	public int[] getShape() {
		return m_index.getShape();
	}

	@Override
	public Class<?> getElementType() {
		Class<?> result = null;
		if( m_n4tdataitem != null ) {
			result = m_n4tdataitem.getDataClass();
		}
		else {
			Object oData = getData();
			if( oData != null )
			{
				if( oData.getClass().isArray() ) {
					result = oData.getClass().getComponentType();
					while( result.isArray() ) {
						result = result.getComponentType();
					}
				}
				else
					result = oData.getClass();
			}
		}
		return result;
	}

    @Override
	public void lock() {
		// TODO Auto-generated method stub
	}

	@Override
	public boolean isDirty() {
		return m_isDirty;
	}

	@Override
	public void releaseStorage() throws BackupException {
		// TODO Auto-generated method stub
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
	public void unlock() {
		// TODO Auto-generated method stub

	}

	/// IArray data manipulation
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
	    NxsIndex index = new NxsIndex( m_shape, reference, range );
        return new NxsArrayIterator(this, index);
	}

	@Override
	public long getRegisterId() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long getSize() {
		long size = m_index.getSize();
		return size;
	}

	@Override
	public ISliceIterator getSliceIterator(int rank)
			throws ShapeNotMatchException, InvalidRangeException {
		return new NxsSliceIterator(this, rank);
	}
    
    // IArray data getters and setters
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
    public Object getObject(IIndex index) {
        return get(index);
    }

    @Override
    public short getShort(IIndex ima) {
        return (( Short ) get(ima)).shortValue();
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
	public IArray setDouble(double value) {
		Object oData = getData();
        if( m_bRawArray ) {
            java.util.Arrays.fill((double[])oData, value);
        }
        else {
            setDouble(oData, value);
        }
        return this;
    }
    
	@Override
	public IArray copy() {
		return copy(true);
	}
	
	@Override
	public IArray copy(boolean data) {
		NxsArray result = new NxsArray(this);
		
		if( data ) {
			result.m_oData = NxsArrayUtils.copyJavaArray(m_oData);
		}
		
		return result;
	}

    @Override
    public void setIndex(IIndex index) {
        m_index = index;
    }
    
    @Override
    public String getFactoryName() {
    	return NxsFactory.NAME;
    }
    
    // ---------------------------------------------------------
	/// Protected methods
	// ---------------------------------------------------------
	protected boolean isSingleRawArray() {
		return m_bRawArray;
	}

    protected IArray sectionNoReduce(int[] origin, int[] shape, long[] stride) throws ShapeNotMatchException {
    	Object oData = getData();
        NxsArray array = new NxsArray(oData, m_shape);
        array.m_index.setShape(shape);
        array.m_index.setStride(stride);
        ((NxsIndex) array.m_index).setOrigin(origin);
        return array;
    }
    
    protected void setShape(int[] shape) {
    	m_shape = shape;
    }
    
    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    /**
     * Translate the given IIndex into the corresponding cell index.
     * This method is used to access a multidimensional array's cell when
     * the memory storage is a single raw array.
     * 
     * @param index sibling a cell in a multidimensional array
     * @return the cell number in a single raw array (that carry the same logical shape)
     */
    private int translateIndex(IIndex index) {
        int[] indexes = index.getCurrentCounter();

        int lPos = 0, lStartRaw;
        for( int k = 1; k < m_shape.length; k++ ) {

            lStartRaw = 1;
            for( int j = 0; j < k; j++ )
            {
                lStartRaw *= m_shape[j];
            }
            lStartRaw *= indexes[k - 1];
            lPos += lStartRaw;
        }
        lPos += indexes[indexes.length - 1];
        return lPos;
    }
    
    /**
     * Get the object targeted by given index and return it (eventually using outboxing).
     * It's the central data access method that all other methods rely on.
     * 
     * @param index targeting a cell 
     * @return the content of cell designed by the index
     * @throws InvalidRangeException if one of the index is bigger than the corresponding dimension shape
     */
    private Object get(IIndex index) {
        Object oCurObj = null;
        NxsIndex idx = null;
        if( index instanceof NxsIndex || index instanceof NxsIndexMatrix ) {
        	idx = (NxsIndex) index;
        }
        else {
        	idx = new NxsIndex(index.getShape(), new int[index.getRank()], index.getShape());
        	idx.set(index.getCurrentCounter());
        }
        Object oData = getData();
        // If it's a string then no array 
        if( oData.getClass().equals( String.class ) )
        {
            return (String) oData;
        }
        // If it's a scalar value then we return it
        else if( ! oData.getClass().isArray() )
        {
            return oData;
        }
        // If it's a single raw array, then we compute indexes to have the corresponding cell number 
        else if( m_bRawArray )
        {
        	int lPos;
        	if( java.util.Arrays.equals(m_n4tdataitem.getStart(), idx.getProjectionOrigin() ) ) {
        		lPos = idx.currentProjectionElement();
        	}
        	else {
        		lPos = (int) idx.currentElement();
        	}
            return java.lang.reflect.Array.get(oData, lPos);
        }
        // If it's a multidimensional array, then we get sub-part until to have the single cell we're interested in
        else
        {
            int[] indexes = idx.getCurrentCounter();
            oCurObj = oData;
            for( int i = 0; i < indexes.length; i++ )
            {
                oCurObj = java.lang.reflect.Array.get(oCurObj, indexes[i]);
            }
        }
        return oCurObj;
    }
    
    private Object getData() {
    	Object result = m_oData;
    	if( result == null && m_n4tdataitem != null ) {
    		result = m_n4tdataitem.getData(((NxsIndex) m_index).getProjectionOrigin(), ((NxsIndex) m_index).getProjectionShape());
    	}
    	return result;
    }
    
    /**
     * Set the given object into the targeted cell by given index (eventually using autoboxing).
     * It's the central data access method that all other methods rely on.
     * 
     * @param index targeting a cell 
     * @param value new value to set in the array
     * @throws InvalidRangeException if one of the index is bigger than the corresponding dimension shape
     */
    private void set(IIndex index, Object value) {
        // If array has string class: then it's a scalar string 
    	Object oData = getData();
        if( oData.getClass().equals( String.class ) )
        {
        	oData = (String) value;
        }
        // If array isn't an array we set the scalar value
        else if( ! oData.getClass().isArray() )
        {
        	oData = value;
        }
        // If it's a single raw array, then we compute indexes to have the corresponding cell number
        else if( m_bRawArray )
        {
            int lPos = translateIndex(index);
            java.lang.reflect.Array.set(oData, lPos, value);
        }
        // Else it's a multidimensional array, so we will take slices from each dimension until we can reach requested cell
        else {
            int[] indexes = null;
            if( index instanceof NxsIndex ) {
            	indexes = ((NxsIndex) index).getCurrentPos();
            }
            else {
            	indexes = index.getCurrentCounter();
            }
            Object oCurObj = oData;
            for( int i = 0; i < indexes.length - 1; i++ )
            {
                oCurObj = java.lang.reflect.Array.get(oCurObj, indexes[i]);
            }
            java.lang.reflect.Array.set(oCurObj, indexes[indexes.length - 1], value);
        }
        
    }
    
    /**
     * Recursive method that sets all values of the given array (whatever it's form is) 
     * to the same given double value
     * 
     * @param array object array to fill
     * @param value double value to be set in the array
     * @return the array filled properly
     * @note ensure the given array is a double[](...[]) or a Double[](...[])
     */
    private Object setDouble(Object array, double value) {
        if( array.getClass().isArray() ) {
            int iLength = java.lang.reflect.Array.getLength(array);
            for (int j = 0; j < iLength; j++) {
                Object o = java.lang.reflect.Array.get(array, j);
                if (o.getClass().isArray()) {
                    setDouble(o, value);
                } else {
                    java.util.Arrays.fill( (double[]) array, value);
                    return array;
                }
            }
        } else
            java.lang.reflect.Array.set(array, 0, value);
        
        return array;
    }

	@Override
	public void setDirty(boolean dirty) {
		// TODO Auto-generated method stub
		
	}
}

