package org.gumtree.data.soleil;

import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IAttribute;

public class NxsAttribute implements IAttribute {

	/// Members
	private String	m_sName;		// Attribute's name
	private IArray	m_aValue;		// Attribute's value

	/// Constructors
	public NxsAttribute(String sName, Object aValue) {
		int i = 1;
		if( aValue.getClass().isArray() )
			i = java.lang.reflect.Array.getLength(aValue);

		m_sName  = sName;
		m_aValue = new NxsArray(aValue, new int[] {i} );
	}


	@Override
	public int getLength() {
		Long length = m_aValue.getSize();
		return length.intValue();
	}

	@Override
	public String getName() {
		return m_sName;
	}

	@Override
	public Number getNumericValue() {
		if( isString() )
		{
			return null;
		}

		if( isArray() )
		{
			return getNumericValue(0);
		}
		else
		{
			return (Number) m_aValue.getStorage();

		}
	}

	@Override
	public Number getNumericValue(int index) {
		Object value;
		if( isArray() )
		{
			value = java.lang.reflect.Array.get(m_aValue.getStorage(), index);
		}
		else
		{
			value = m_aValue.getStorage();
		}

		if( isString() )
			return (Double) value;

		return (Number) value;
	}

	@Override
	public String getStringValue() {
		if( isString() )
		{
			return (String) m_aValue.getStorage();
		}
		else
		{
			return null;
		}
	}

	@Override
	public String getStringValue(int index) {
		if( isString() )
		{
			return ((String) java.lang.reflect.Array.get(m_aValue.getStorage(), index));
		}
		else
		{
			return null;
		}
	}

	@Override
	public Class<?> getType() {
		return m_aValue.getElementType();
	}

	@Override
	public IArray getValue() {
		return m_aValue;
	}

	@Override
	public boolean isArray() {
		return m_aValue.getStorage().getClass().isArray();
	}

	@Override
	public boolean isString() {
		Class<?> tmpClass = "".getClass();
		return ( m_aValue.getElementType().equals(tmpClass) );
	}

	@Override
	public void setStringValue(String val) {
		m_aValue = new NxsArray(val, new int[] {1});
	}

	@Override
	public void setValue(IArray value) {
		m_aValue = value;
	}
    
    public String toString() {
        return m_sName + "=" + m_aValue;
    }

	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}
}
