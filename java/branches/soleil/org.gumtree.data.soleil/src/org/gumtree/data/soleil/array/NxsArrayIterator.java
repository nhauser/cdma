package org.gumtree.data.soleil.array;

import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IArrayIterator;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.soleil.NxsFactory;

public class NxsArrayIterator implements IArrayIterator {
	/// Members
	private IArray  m_array;
    private IIndex  m_index;
    private Object  m_current;
    private boolean m_access;

	public NxsArrayIterator(NxsArrayInterface array)
	{
		m_array	= array;
		// [ANSTO][Tony][2011-08-31] Should m_access set to true for NxsArrayInterface??
		// If m_access is set to false, next() does not work. 
		m_access = true;
		try {
			m_index = array.getIndex().clone();
			m_index.set( new int[m_index.getRank()] );
			m_current = m_array.getObject(m_index);
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
	}
    
    public NxsArrayIterator(IArray array, IIndex index) {
        this(array, index, true);
    }
    
    public NxsArrayIterator(IArray array, IIndex index, boolean accessData) {
        m_array   = array;
        m_index   = index;
        m_access  = accessData;
        if( m_access ) {
        	m_current = m_array.getObject(m_index);
        }
    }

	@Override
	public boolean getBooleanCurrent() {
		return ((Boolean) m_current).booleanValue();
	}

	@Override
	public boolean getBooleanNext() {
		return ((Boolean) next());
	}

	@Override
	public byte getByteCurrent() {
		return ((Byte) m_current).byteValue();
	}

	@Override
	public byte getByteNext() {
        return ((Byte) next()).byteValue();
	}

	@Override
	public char getCharCurrent() {
		return ((Character) m_current).charValue();
	}

	@Override
	public char getCharNext() {
		return ((Character) next()).charValue();
	}

	@Override
	public int[] getCurrentCounter() {
        return m_index.getCurrentCounter();
	}

	@Override
	public double getDoubleCurrent() {
		return ((Number) m_current).doubleValue();
	}

	@Override
	public double getDoubleNext() {
		return ((Number) next()).doubleValue();
	}

	@Override
	public float getFloatCurrent() {
		return ((Number) m_current).floatValue();
	}

	@Override
	public float getFloatNext()	{
		return ((Number) next()).floatValue();
	}

	@Override
	public int getIntCurrent() {
		return ((Number) m_current).intValue();
	}

	@Override
	public int getIntNext() {
		return ((Number) next()).intValue();
	}

	@Override
	public long getLongCurrent() {
		return ((Number) m_current).longValue();
	}

	@Override
	public long getLongNext() {
		return ((Number) next()).longValue();
	}

	@Override
	public Object getObjectCurrent() {
        return m_current;
	}

	@Override
	public Object getObjectNext() {
		return next();
	}

	@Override
	public short getShortCurrent() {
		return ((Number) m_current).shortValue();
	}

	@Override
	public short getShortNext() {
		return ((Number) next()).shortValue();
	}

	@Override
	public boolean hasNext()
	{
        long index = m_index.currentElement();
        long last  = m_index.lastElement();
        return ( index <= last && index >= 0);
	}
	
	@Override
	public boolean hasCurrent() {
		return ( this.getObjectCurrent() != null );
	}

	@Override
	public Object next()
	{
		long currentPos = m_index.currentElement();
		if( m_access ) {
	    	if( currentPos <= m_index.lastElement() && currentPos != -1 ) {
	    		m_current = m_array.getObject(m_index);
	    	}
	    	else {
	    		m_current = null;
	    	}
		}
    	incrementIndex(m_index);
		return m_current;
	}

	@Override
	public void setBooleanCurrent(boolean val) {
    	setObjectCurrent(val);
	}

	@Override
	public void setBooleanNext(boolean val) {
        setObjectNext(val);
	}

	@Override
	public void setByteCurrent(byte val) {
        setObjectCurrent(val);
	}

	@Override
	public void setByteNext(byte val) {
        setObjectNext(val);
	}

	@Override
	public void setCharCurrent(char val) {
        setObjectCurrent(val);
    }

	@Override
	public void setCharNext(char val) {
        setObjectNext(val);
	}

	@Override
	public void setDoubleCurrent(double val) {
        setObjectCurrent(val);
	}

	@Override
	public void setDoubleNext(double val) {
        setObjectNext(val);
    }

	@Override
	public void setFloatCurrent(float val) {
        setObjectCurrent(val);
	}

	@Override
	public void setFloatNext(float val) {
        setObjectNext(val);
	}

	@Override
	public void setIntCurrent(int val) {
        setObjectCurrent(val);
	}

	@Override
	public void setIntNext(int val) {
        setObjectNext(val);
	}

	@Override
	public void setLongCurrent(long val) {
        setObjectCurrent(val);
	}

	@Override
	public void setLongNext(long val) {
        setObjectNext(val);
	}

	@Override
	public void setObjectCurrent(Object val) {
		m_current = val;
        m_array.setObject(m_index, val);
	}

	@Override
	public void setObjectNext(Object val) {
		// [ANSTO][Tony][2011-08-31] Index should be incremented after object is set
        setObjectCurrent(val);
	    incrementIndex(m_index);
	}

	@Override
	public void setShortCurrent(short val) {
        setObjectCurrent(val);
	}

	@Override
	public void setShortNext(short val) {
        setObjectNext(val);
    }

	static public void incrementIndex(IIndex index)
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
		return NxsFactory.NAME;
	}
	
	/// protected method
	protected void incrementIndex() {
		NxsArrayIterator.incrementIndex(m_index);
	}
}