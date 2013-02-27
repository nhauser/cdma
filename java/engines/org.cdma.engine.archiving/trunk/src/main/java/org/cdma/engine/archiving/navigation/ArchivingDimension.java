package org.cdma.engine.archiving.navigation;

import org.cdma.exception.NotImplementedException;
import org.cdma.exception.ShapeNotMatchException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IDimension;

public class ArchivingDimension implements IDimension {
	private IArray mArray;
	private String mName;
	private String mFactory;

	public ArchivingDimension( String factory, IArray array, String name ) {
		mFactory = factory;
		mName = name;
		mArray = array;
	}
	
	@Override
	public String getFactoryName() {
		return mFactory;
	}

	@Override
	public String getName() {
		return mName;
	}

	@Override
	public int getLength() {
		return mArray != null ? ((Long) mArray.getSize()).intValue() : -1;
	}

	@Override
	public boolean isUnlimited() {
		return false;
	}

	@Override
	public boolean isVariableLength() {
		return false;
	}

	@Override
	public boolean isShared() {
		return false;
	}

	@Override
	public IArray getCoordinateVariable() {
		return mArray;
	}

	@Override
	public int compareTo(Object o) {
		int result = -1;
		if( o instanceof IDimension ) {
			result = mName.compareTo( ((IDimension) o).getName() );
		}
		return result;
	}

	@Override
	public String writeCDL(boolean strict) {
		throw new NotImplementedException();
	}

	@Override
	public void setUnlimited(boolean b) {
		throw new NotImplementedException();

	}

	@Override
	public void setVariableLength(boolean b) {
		throw new NotImplementedException();

	}

	@Override
	public void setShared(boolean b) {
		throw new NotImplementedException();

	}

	@Override
	public void setLength(int n) {
		throw new NotImplementedException();

	}

	@Override
	public void setName(String name) {
		mName = name;
	}

	@Override
	public void setCoordinateVariable(IArray array) throws ShapeNotMatchException {
		mArray = array;
	}
}
