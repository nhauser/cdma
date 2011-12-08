package org.gumtree.data.engine.jnexus.array;

import org.gumtree.data.engine.jnexus.NexusFactory;
import org.gumtree.data.engine.jnexus.utils.NexusArrayMath;
import org.gumtree.data.engine.jnexus.utils.NexusArrayUtils;
import org.gumtree.data.exception.BackupException;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IArrayIterator;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.interfaces.ISliceIterator;
import org.gumtree.data.math.IArrayMath;
import org.gumtree.data.utils.IArrayUtils;

import fr.soleil.nexus4tango.DataItem;


public class NexusArray implements IArray {
    private IIndex    mIndex;        // IIndex corresponding to this Array (dimension sizes defining the viewable part of the array)
	private Object	  mData;        // It's an array of values
	private boolean	  mIsRawArray;    // True if the stored array has a rank of 1 (independently of its shape)
    private boolean   mIsDirty;      // Is the array synchronized with the handled file
    private DataItem  mN4TDataItem;  // Array of datasets that are used to store the storage backing
    private int[]     mShape;        // Shape of the array (dimension sizes of the storage backing) 
    
	// Constructors
	public NexusArray(Object oArray, int[] iShape) {
        mIndex = new NexusIndex(iShape);
		mData  = oArray;
		mShape = iShape;
        if( iShape.length == 1 ) {
            mIsRawArray	= true;
        }
        else {
            mIsRawArray = (
                            oArray.getClass().isArray() && 
                            !(java.lang.reflect.Array.get(oArray, 0).getClass().isArray()) 
                          );
        }
        mN4TDataItem = null;
	}

    public NexusArray(NexusArray array) {
        try {
			mIndex = array.mIndex.clone();
		} catch (CloneNotSupportedException e) {
			mIndex = null;
		}
        mData  = array.mData;
        mShape = new int[array.mShape.length];
        int i  = 0;
        for( int size : array.mShape ) {
            mShape[i++]  = size;
        }
        mIsRawArray = array.mIsRawArray;
        mIsDirty    = array.mIsDirty;
        try {
			mN4TDataItem = array.mN4TDataItem.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
    }
    
    public NexusArray(DataItem ds) {
        mIndex       = new NexusIndex(ds);
		mData        = null;
		mShape       = ds.getSize();
		mIsRawArray  = ds.isSingleRawArray();
        mIsDirty     = false;
        mN4TDataItem = ds;
	}
    
	// ---------------------------------------------------------
	/// public methods
	// ---------------------------------------------------------
    @Override
    public IArrayMath getArrayMath() {
        return new NexusArrayMath(this);
    }

    @Override
    public IArrayUtils getArrayUtils() {
        return new NexusArrayUtils(this);
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
        Object data = null;
	    while (ii.hasNext())
	    {
            data = ii.next();
            sbuff.append(data);
            sbuff.append(" ");
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
		return mIndex.getShape();
	}

	@Override
	public Class<?> getElementType() {
		Class<?> result = null;
		if( mN4TDataItem != null ) {
			result = mN4TDataItem.getDataClass();
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
		return mIsDirty;
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
		return mIndex;
	}

	@Override
	public IArrayIterator getIterator() {
		return (IArrayIterator) new NexusArrayIterator(this);
	}

	@Override
	public int getRank() {
		return mIndex.getRank();
	}

	@Override
	public IArrayIterator getRegionIterator(int[] reference, int[] range)
			throws InvalidRangeException {
	    NexusIndex index = new NexusIndex( mShape, reference, range );
        return new NexusArrayIterator(this, index);
	}

	@Override
	public long getRegisterId() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long getSize() {
		long size = mIndex.getSize();
		return size;
	}

	@Override
	public ISliceIterator getSliceIterator(int rank)
			throws ShapeNotMatchException, InvalidRangeException {
		return new NexusSliceIterator(this, rank);
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
        if( mIsRawArray ) {
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
		NexusArray result = new NexusArray(this);
		
		if( data ) {
			result.mData = NexusArrayUtils.copyJavaArray(mData);
		}
		
		return result;
	}

    @Override
    public void setIndex(IIndex index) {
        mIndex = index;
    }
    
    @Override
    public String getFactoryName() {
    	return NexusFactory.NAME;
    }
    
    // ---------------------------------------------------------
	/// Protected methods
	// ---------------------------------------------------------
	protected boolean isSingleRawArray() {
		return mIsRawArray;
	}

    protected IArray sectionNoReduce(int[] origin, int[] shape, long[] stride) throws ShapeNotMatchException {
    	Object oData = getData();
        NexusArray array = new NexusArray(oData, mShape);
        array.mIndex.setShape(shape);
        array.mIndex.setStride(stride);
        ((NexusIndex) array.mIndex).setOrigin(origin);
        return array;
    }
    
    protected void setShape(int[] shape) {
    	mShape = shape;
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
        for( int k = 1; k < mShape.length; k++ ) {

            lStartRaw = 1;
            for( int j = 0; j < k; j++ )
            {
                lStartRaw *= mShape[j];
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
        NexusIndex idx = null;
        if( index instanceof NexusIndex ) {
        	idx = (NexusIndex) index;
        }
        else {
        	idx = new NexusIndex(index.getShape(), new int[index.getRank()], index.getShape());
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
        else if( mIsRawArray )
        {
        	int lPos;
        	if( java.util.Arrays.equals(mN4TDataItem.getStart(), idx.getProjectionOrigin() ) ) {
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
    	Object result = mData;
    	if( result == null && mN4TDataItem != null ) {
    		result = mN4TDataItem.getData(((NexusIndex) mIndex).getProjectionOrigin(), ((NexusIndex) mIndex).getProjectionShape());
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
        else if( mIsRawArray )
        {
            int lPos = translateIndex(index);
            java.lang.reflect.Array.set(oData, lPos, value);
        }
        // Else it's a multidimensional array, so we will take slices from each dimension until we can reach requested cell
        else {
            int[] indexes = null;
            if( index instanceof NexusIndex ) {
            	indexes = ((NexusIndex) index).getCurrentPos();
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

