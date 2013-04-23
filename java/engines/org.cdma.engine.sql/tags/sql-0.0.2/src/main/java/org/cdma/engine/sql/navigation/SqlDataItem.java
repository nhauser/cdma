//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.sql.navigation;

import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.sql.array.SqlArray;
import org.cdma.exception.DimensionNotSupportedException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IRange;
import org.cdma.utils.Utilities.ModelType;

public class SqlDataItem implements IDataItem {
	private IDataset mDataset;
	private String   mFactory;
	private SqlGroup mParent;
	private String   mName;
	private IArray   mArray;

	public SqlDataItem(String factory, SqlGroup parent, String name, ResultSet set, int column) {
		this( factory, parent, name );
		init( set, column, -1 );
	}
	
	public SqlDataItem(String factory, SqlGroup parent, String name) {
		mFactory = factory;
		mParent  = parent;
		mDataset = parent != null ? parent.getDataset() : null;
		mName    = name;
	}

	protected void init( ResultSet set, int column, int rows ) {
		try {
			mArray = SqlArray.instantiate(mFactory, set, column, rows, null);
			((SqlArray) mArray).appendData(set);
		} catch( SQLException e ) {
			mArray = null;
			Factory.getLogger().log(Level.SEVERE, "Unable to initialize data item's value!", e);
		}
	}
	
	@Override
	public ModelType getModelType() {
		return ModelType.DataItem;
	}

	@Override
	public void addOneAttribute(IAttribute attribute) {
		throw new NotImplementedException();

	}

	@Override
	public void addStringAttribute(String name, String value) {
		throw new NotImplementedException();

	}

	@Override
	public IAttribute getAttribute(String name) {
		throw new NotImplementedException();
	}

	@Override
	public List<IAttribute> getAttributeList() {
		throw new NotImplementedException();
	}

	@Override
	public IDataset getDataset() {
		return mDataset;
	}

	@Override
	public String getLocation() {
		throw new NotImplementedException();
	}

	@Override
	public String getName() {
		return mParent.getName() + "/" + getShortName();
	}

	@Override
	public IContainer getParentGroup() {
		return mParent;
	}

	@Override
	public IContainer getRootGroup() {
		return mDataset.getRootGroup();
	}

	@Override
	public String getShortName() {
		return mName;
	}

	@Override
	public boolean hasAttribute(String name, String value) {
		throw new NotImplementedException();
	}

	@Override
	public void setName(String name) {
		throw new NotImplementedException();
	}

	@Override
	public void setShortName(String name) {
		mName = name;
	}

	@Override
	public void setParent(IGroup group) {
		if( group instanceof SqlGroup ) {
			mParent = (SqlGroup) group;
		}
	}

	@Override
	public long getLastModificationDate() {
		throw new NotImplementedException();
	}

	@Override
	public String getFactoryName() {
		return mFactory;
	}

	@Override
	public IAttribute findAttributeIgnoreCase(String name) {
		throw new NotImplementedException();
	}

	@Override
	public int findDimensionIndex(String name) {
		throw new NotImplementedException();
	}

	@Override
	public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
		throw new NotImplementedException();
	}

	@Override
	public IArray getData() throws IOException {
		return mArray;
	}

	@Override
	public IArray getData(int[] origin, int[] shape) throws IOException, InvalidRangeException {
		return mArray.getArrayUtils().section(origin, shape).getArray();
	}

	@Override
	public String getDescription() {
		throw new NotImplementedException();
	}

	@Override
	public List<IDimension> getDimensions(int index) {
		throw new NotImplementedException();
	}

	@Override
	public List<IDimension> getDimensionList() {
		throw new NotImplementedException();
	}

	@Override
	public String getDimensionsString() {
		throw new NotImplementedException();
	}

	@Override
	public int getElementSize() {
		throw new NotImplementedException();
	}

	@Override
	public String getNameAndDimensions() {
		throw new NotImplementedException();
	}

	@Override
	public void getNameAndDimensions(StringBuffer buf, boolean longName, boolean length) {
		throw new NotImplementedException();

	}

	@Override
	public List<IRange> getRangeList() {
		throw new NotImplementedException();
	}

	@Override
	public int getRank() {
		return mArray.getRank();
	}

	@Override
	public IDataItem getSection(List<IRange> section) throws InvalidRangeException {
		throw new NotImplementedException();
	}

	@Override
	public List<IRange> getSectionRanges() {
		throw new NotImplementedException();
	}

	@Override
	public int[] getShape() {
		return mArray.getShape();
	}

	@Override
	public long getSize() {
		return mArray.getSize();
	}

	@Override
	public int getSizeToCache() {
		throw new NotImplementedException();
	}

	@Override
	public IDataItem getSlice(int dim, int value) throws InvalidRangeException {
		throw new NotImplementedException();
	}

	@Override
	public Class<?> getType() {
		return mArray.getElementType();
	}

	@Override
	public String getUnitsString() {
		throw new NotImplementedException();
	}

	@Override
	public boolean hasCachedData() {
		throw new NotImplementedException();
	}

	@Override
	public void invalidateCache() {
		throw new NotImplementedException();

	}

	@Override
	public boolean isCaching() {
		throw new NotImplementedException();
	}

	@Override
	public boolean isMemberOfStructure() {
		throw new NotImplementedException();
	}

	@Override
	public boolean isMetadata() {
		throw new NotImplementedException();
	}

	@Override
	public boolean isScalar() {
		return ( getRank() == 0 );
	}

	@Override
	public boolean isUnlimited() {
		throw new NotImplementedException();
	}

	@Override
	public boolean isUnsigned() {
		throw new NotImplementedException();
	}

	@Override
	public byte readScalarByte() throws IOException {
		return mArray.getByte( mArray.getIndex() );
	}

	@Override
	public double readScalarDouble() throws IOException {
		return mArray.getDouble( mArray.getIndex() );
	}

	@Override
	public float readScalarFloat() throws IOException {
		return mArray.getFloat( mArray.getIndex() );
	}

	@Override
	public int readScalarInt() throws IOException {
		return mArray.getInt( mArray.getIndex() );
	}

	@Override
	public long readScalarLong() throws IOException {
		return mArray.getLong( mArray.getIndex() );
	}

	@Override
	public short readScalarShort() throws IOException {
		return mArray.getShort( mArray.getIndex() );
	}

	@Override
	public String readScalarString() throws IOException {
		String result = null;
		if( mArray.getElementType().equals(String.class) ) {
			result = (String) mArray.getObject( mArray.getIndex() );
		}
		return result;
	}

	@Override
	public boolean removeAttribute(IAttribute a) {
		throw new NotImplementedException();
	}

	@Override
	public void setCachedData(IArray cacheData, boolean isMetadata) throws InvalidArrayTypeException {
		mArray = cacheData;
	}

	@Override
	public void setCaching(boolean caching) {
		throw new NotImplementedException();
	}

	@Override
	public void setDataType(Class<?> dataType) {
		throw new NotImplementedException();
	}

	@Override
	public void setDimensions(String dimString) {
		throw new NotImplementedException();
	}

	@Override
	public void setDimension(IDimension dim, int ind) throws DimensionNotSupportedException {
		throw new NotImplementedException();
	}

	@Override
	public void setElementSize(int elementSize) {
		throw new NotImplementedException();
	}

	@Override
	public void setSizeToCache(int sizeToCache) {
		throw new NotImplementedException();

	}

	@Override
	public void setUnitsString(String units) {
		throw new NotImplementedException();

	}

	@Override
	public String toStringDebug() {
		throw new NotImplementedException();
	}

	@Override
	public String writeCDL(String indent, boolean useFullName, boolean strict) {
		throw new NotImplementedException();
	}
	
	@Override
	public IDataItem clone() {
		SqlDataItem item = new SqlDataItem(mFactory, mParent, mName);
		item.mArray = mArray.copy();
		return item;
	}
}
