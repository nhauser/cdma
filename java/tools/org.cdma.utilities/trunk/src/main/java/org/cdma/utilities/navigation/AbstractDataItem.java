//******************************************************************************
// Copyright (c) 2013 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// See AUTHORS file
//******************************************************************************
package org.cdma.utilities.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.cdma.exception.BackupException;
import org.cdma.exception.DimensionNotSupportedException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IRange;
import org.cdma.utilities.navigation.internal.NodeParentAttribute;
import org.cdma.utils.Utilities.ModelType;

public abstract class AbstractDataItem extends NodeParentAttribute implements IDataItem {
	private IArray mData;
	private Map<String, Integer> mDimensions;
	
	public AbstractDataItem(String factory, IDataset dataset, IGroup parent, String name, IArray data) throws BackupException {
		super(factory, dataset, parent, name);
		mData = data;
		mDimensions = new HashMap<String, Integer>();
	}

	// ------------------------------------------------------------------------
	// Abstract methods
	// ------------------------------------------------------------------------
	@Override
	abstract public long getLastModificationDate();
	
	abstract public IDataItem clone();
	
	// ------------------------------------------------------------------------
	// Implemented methods
	// ------------------------------------------------------------------------
	@Override
	public ModelType getModelType() {
		return ModelType.DataItem;
	}

	@Override
	public IAttribute findAttributeIgnoreCase(String name) {
		IAttribute result = null;
		if( name != null ) {
			List<IAttribute> attributes = getAttributeList();
			for( IAttribute attribute : attributes ) {
				if( name.equalsIgnoreCase( attribute.getName() ) ) {
					result = attribute;
				}
			}
		}
		return result;
	}

	@Override
	public int findDimensionIndex(String name) {
		int index = -1;
		if( name != null ) {
			index = mDimensions.get(name);
		}
		return index;
	}

	@Override
	public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IArray getData() throws IOException {
		return mData;
	}

	@Override
	public IArray getData(int[] origin, int[] shape) throws IOException, InvalidRangeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getDescription() {
		String result = null;
		IAttribute desc = getAttribute("description");
		if( desc != null && desc.isString() && !desc.isArray() ) {
			result = desc.getStringValue();
		}
		return result;
	}

	@Override
	public List<IDimension> getDimensions(int index) {
		List<IDimension> result = new ArrayList<IDimension>();
		IGroup parent = getParentGroup();
		IDimension dim;
		if( parent != null ) {
			for( Entry<String, Integer> entry : mDimensions.entrySet() ) {
				if( index == entry.getValue() ) {
					dim = parent.getDimension( entry.getKey() );
					if( dim != null ) {
						result.add( dim );
					}
				}
			}
		}
		return result;
	}

	@Override
	public List<IDimension> getDimensionList() {
		List<IDimension> result = new ArrayList<IDimension>();
		IGroup parent = getParentGroup();
		if( parent != null ) {
			IDimension dim;
			for( Entry<String, Integer> entry : mDimensions.entrySet() ) {
				dim = parent.getDimension( entry.getKey() );
				if( dim != null ) {
					result.add( dim );
				}
			}
		}
		return result;
	}

	@Override
	public String getDimensionsString() {
		StringBuffer result = new StringBuffer();
		getNameAndDimensions( result, false, false);
		return result.toString();
	}

	@Override
	public int getElementSize() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String getNameAndDimensions() {
		StringBuffer result = new StringBuffer();
		getNameAndDimensions( result, true, false);
		return result.toString();
	}

	@Override
	public void getNameAndDimensions(StringBuffer buf, boolean longName, boolean length) {
		if( longName ) {
			buf.append( getName() );
			buf.append( " " );
		}
		
		IGroup parent = getParentGroup();
		int i = 0;
		for( String name : mDimensions.keySet() ) {
			if( i != 0 ) {
				buf.append( " " );
			}
			buf.append( name );
			if( length && parent != null ) {
				buf.append(":" );
				buf.append( parent.getDimension(name).getLength() );
			}
			i++;
		}
	}

	@Override
	public List<IRange> getRangeList() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getRank() {
		int result = -1;
		try {
			IArray data = getData();
			if( data != null ) {
				result = data.getRank();
			}
		} catch (IOException e) {
		}
		return result;
	}

	@Override
	public IDataItem getSection(List<IRange> section)
			throws InvalidRangeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<IRange> getSectionRanges() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int[] getShape() {
		int[] result = null;
		try {
			IArray data = getData();
			if( data != null ) {
				result = data.getShape();
			}
		} catch (IOException e) {
		}
		return result;
	}

	@Override
	public long getSize() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getSizeToCache() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public IDataItem getSlice(int dim, int value) throws InvalidRangeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Class<?> getType() {
		Class<?> clazz = null;
		if( mData != null ) {
			clazz = mData.getElementType();
		}
		return clazz;
	}

	@Override
	public String getUnitsString() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean hasCachedData() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void invalidateCache() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean isCaching() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isMemberOfStructure() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isMetadata() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isScalar() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isUnlimited() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isUnsigned() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public byte readScalarByte() throws IOException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double readScalarDouble() throws IOException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float readScalarFloat() throws IOException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int readScalarInt() throws IOException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long readScalarLong() throws IOException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public short readScalarShort() throws IOException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String readScalarString() throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setCachedData(IArray cacheData, boolean isMetadata) throws InvalidArrayTypeException {
		mData = cacheData;
	}

	@Override
	public void setCaching(boolean caching) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setDataType(Class<?> dataType) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setDimensions(String dimString) {
		// TODO
		throw new NotImplementedException();
		
	}

	@Override
	public void setDimension(IDimension dim, int ind) throws DimensionNotSupportedException {
		if( dim == null ) {
			throw new DimensionNotSupportedException("A null dimension can't be affected!");
		}
		mDimensions.put(dim.getName(), ind);
		
	}

	@Override
	public void setElementSize(int elementSize) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setSizeToCache(int sizeToCache) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setUnitsString(String units) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String toStringDebug() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String writeCDL(String indent, boolean useFullName, boolean strict) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected String getPathSeparator() {
		// TODO Auto-generated method stub
		return null;
	}
	
	

	
}
