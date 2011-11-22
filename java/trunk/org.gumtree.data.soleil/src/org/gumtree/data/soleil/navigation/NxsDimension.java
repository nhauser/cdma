package org.gumtree.data.soleil.navigation;

import java.io.IOException;

import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDimension;
import org.gumtree.data.soleil.NxsFactory;

public class NxsDimension implements IDimension {

    private IArray    m_array;
    private String    m_longName;
    private boolean   m_variableLength;
    private boolean   m_unlimited;
    private boolean   m_shared;
    
    public NxsDimension(IDataItem item) {
        m_longName  = item.getName();
        m_unlimited = item.isUnlimited();
        try {
            m_array = item.getData();
        } catch( IOException e ) {
            m_array = null;
        }
    }
    
    public NxsDimension(NxsDimension dim) {
        m_longName       = dim.m_longName;
        m_array          = dim.m_array;
        m_variableLength = dim.m_variableLength;
        m_unlimited      = dim.m_unlimited;
    }
    
	@Override
	public int compareTo(Object o) {
		IDimension dim = (IDimension) o;
		return m_longName.compareTo( dim.getName() );
	}

	@Override
	public IArray getCoordinateVariable() {
		return m_array;
	}

	@Override
	public int getLength() {
        return new Long(m_array.getSize()).intValue();
	}

	@Override
	public String getName() {
		return m_longName;
	}

	@Override
	public boolean isShared() {
		return m_shared;
	}

	@Override
	public boolean isUnlimited() {
		return m_unlimited;
	}

	@Override
	public boolean isVariableLength() {
		return m_variableLength;
	}

	@Override
	public void setLength(int n) {
	    try {
	        m_array.getArrayUtils().reshape( new int[] {n} );
        } catch ( ShapeNotMatchException e) {
            e.printStackTrace();
        }
    }

	@Override
	public void setName(String name) {
        m_longName = name;
	}

	@Override
	public void setShared(boolean b) {
        m_shared = b;
	}

	@Override
	public void setUnlimited(boolean b) {
	    m_unlimited = b;
	}

	@Override
	public void setVariableLength(boolean b) {
	    m_variableLength = b;
	}
    
    @Override
    public void setCoordinateVariable(IArray array) throws ShapeNotMatchException {
        if( java.util.Arrays.equals(m_array.getShape(), array.getShape()) )
            throw new ShapeNotMatchException("Arrays must have same shape!");
        m_array = array;
    }

	@Override
	public String writeCDL(boolean strict) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}
}
