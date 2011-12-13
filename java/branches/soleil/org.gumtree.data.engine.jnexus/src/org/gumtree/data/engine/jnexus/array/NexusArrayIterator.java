package org.gumtree.data.engine.jnexus.array;

import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IArrayIterator;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.engine.jnexus.NexusFactory;

public final class NexusArrayIterator implements IArrayIterator {
	/// Members
	private IArray  mArray;
    private IIndex  mIndex;
    private Object  mCurrent;
    private boolean mAccess; // Indicates that this can access the storage memory or not

	public NexusArrayIterator(NexusArray array)
	{
		mArray	= array;
		// [ANSTO][Tony][2011-08-31] Should mAccess set to true for NxsArrayInterface??
		// If m_access is set to false, next() does not work.
		// [SOLEIL][Clement][2011-11-22] Yes it should. It indicates that the iterator shouldn't access memory. In case of hudge matrix the next() will update m_current (i.e. value), but the underlying NeXus engine will automatically load the part corresponding to the view defined by this iterator, which can lead to java heap space memory exception (see NxsArray : private Object getData() ) 
		mAccess = true;
		try {
			mIndex = array.getIndex().clone();
			mIndex.set( new int[mIndex.getRank()] );
			mIndex.setDim( mIndex.getRank() - 1, -1);
		} catch (CloneNotSupportedException e) {
		}
	}
    
    public NexusArrayIterator(IArray array, IIndex index) {
        this(array, index, true);
    }
    
    public NexusArrayIterator(IArray array, IIndex index, boolean accessData) {
    	int[] count = index.getCurrentCounter();
        mArray     = array;
        mIndex     = index;
        mAccess    = accessData;
        count[mIndex.getRank() - 1]--;
        mIndex.set( count );
    }

	@Override
	public boolean getBooleanNext() {
		return ((Boolean) next());
	}

	@Override
	public byte getByteNext() {
        return ((Byte) next()).byteValue();
	}

	@Override
	public char getCharNext() {
		return ((Character) next()).charValue();
	}

	@Override
	public int[] getCounter() {
        return mIndex.getCurrentCounter();
	}

	@Override
	public double getDoubleNext() {
		return ((Number) next()).doubleValue();
	}

	@Override
	public float getFloatNext()	{
		return ((Number) next()).floatValue();
	}

	@Override
	public int getIntNext() {
		return ((Number) next()).intValue();
	}

	@Override
	public long getLongNext() {
		return ((Number) next()).longValue();
	}

	@Override
	public Object getObjectNext() {
		return next();
	}

	@Override
	public short getShortNext() {
		return ((Number) next()).shortValue();
	}

	@Override
	public boolean hasNext()
	{
        long index = mIndex.currentElement();
        long last  = mIndex.lastElement();
        return ( index < last && index >= -1);
	}
	
	@Override
	public Object next()
	{
		incrementIndex(mIndex);
		if( mAccess ) {
			long currentPos = mIndex.currentElement();
	    	if( currentPos <= mIndex.lastElement() && currentPos != -1 ) {
	    		mCurrent = mArray.getObject(mIndex);
	    	}
	    	else {
	    		mCurrent = null;
	    	}
		}
		return mCurrent;
	}

	@Override
	public void setBoolean(boolean val) {
    	setObject(val);
	}

	@Override
	public void setByte(byte val) {
        setObject(val);
	}

	@Override
	public void setChar(char val) {
        setObject(val);
    }

	@Override
	public void setDouble(double val) {
        setObject(val);
	}

	@Override
	public void setFloat(float val) {
        setObject(val);
	}

	@Override
	public void setInt(int val) {
        setObject(val);
	}

	@Override
	public void setLong(long val) {
        setObject(val);
	}

	@Override
	public void setObject(Object val) {
		mCurrent = val;
        mArray.setObject(mIndex, val);
	}

	@Override
	public void setShort(short val) {
        setObject(val);
	}

	public static void incrementIndex(IIndex index)
	{
        int[] current = index.getCurrentCounter();
        int[] shape = index.getShape();
		for( int i = current.length - 1; i >= 0; i-- )
		{
            if( current[i] + 1 >= shape[i] && i > 0)
			{
            	current[i] = 0;
			}
			else
			{
				current[i]++;
				index.set(current);
				return;
			}
		}
	}

	@Override
	public String getFactoryName() {
		return NexusFactory.NAME;
	}
	
	/// protected method
	protected void incrementIndex() {
		NexusArrayIterator.incrementIndex(mIndex);
	}

	/**
	 * @deprecated
	 */
	@Override
	public boolean hasCurrent() {
		return false;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setDoubleNext(double val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public double getDoubleCurrent() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setDoubleCurrent(double val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setFloatNext(float val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public float getFloatCurrent() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setFloatCurrent(float val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setLongNext(long val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public long getLongCurrent() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setLongCurrent(long val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setIntNext(int val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public int getIntCurrent() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setIntCurrent(int val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setShortNext(short val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public short getShortCurrent() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @deprecated
	 */
	@Override
	public short getShort() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setShortCurrent(short val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setByteNext(byte val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public byte getByteCurrent() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setByteCurrent(byte val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setCharNext(char val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public char getCharCurrent() {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setCharCurrent(char val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setBooleanNext(boolean val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public boolean getBooleanCurrent() {
		// TODO Auto-generated method stub
		return false;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setBooleanCurrent(boolean val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setObjectNext(Object val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public Object getObjectCurrent() {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * @deprecated
	 */
	@Override
	public void setObjectCurrent(Object val) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @deprecated
	 */
	@Override
	public int[] getCurrentCounter() {
		// TODO Auto-generated method stub
		return new int[0];
	}
}