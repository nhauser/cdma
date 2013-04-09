package org.cdma.engine.archiving.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.cdma.engine.archiving.internal.attribute.AttributePath;
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
import org.cdma.utils.IArrayUtils;
import org.cdma.utils.Utilities.ModelType;

public class ArchivingDataItem implements IDataItem {

	private String mName;
	private IArray mData;
	private IGroup mParent;
	private String mFactory;
	private Map<String, IAttribute> mAttributes;

	// Constructor
	public ArchivingDataItem(String factory, String name, IGroup parent, IArray data) {
		mFactory = factory;
		mName = name;
		mData = data;
		mParent = parent;
		mAttributes = new HashMap<String, IAttribute>();
	}
	
	@Override
	public ArchivingDataItem clone() {
		ArchivingDataItem clone = new ArchivingDataItem( mFactory, mName, mParent, mData);
		for ( Entry<String, IAttribute> entry : mAttributes.entrySet() ) {
			clone.mAttributes.put(entry.getKey(), entry.getValue() );
		}
		return clone;
	}

	public void setAttributeData(String attributeName) {
		mName = attributeName;
	}

	@Override
	public ModelType getModelType() {
		return ModelType.DataItem;
	}

	@Override
	public IAttribute findAttributeIgnoreCase(String name) {
		IAttribute result = null;
		List<IAttribute> attrList = getAttributeList();
		for (IAttribute attribute : attrList) {
			if (name.equalsIgnoreCase(attribute.getName())) {
				result = attribute;
				break;
			}
		}
		return result;
	}

	@Override
	public int findDimensionIndex(String name) {
		int index = -1;
		IGroup parent = getParentGroup();
		
		if( parent != null && name != null ) {
			int i = 0;
			for( IDimension dimension : parent.getDimensionList() ) {
				if( name.equalsIgnoreCase( dimension.getName() ) ) {
					index = i;
				}
				i++;
			}
		}
		
		return index;
	}

	@Override
	public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
		IDataItem result = this.clone();
		
		try {
			IArray array = result.getData();
			IArrayUtils util = array.getArrayUtils().slice(dimension, value);
			result.setCachedData( util.getArray(), false );
		} catch (IOException e) {
			throw new InvalidRangeException(e);
		} catch (InvalidArrayTypeException e) {
			throw new InvalidRangeException(e);
		}
		
		return result;
	}

	@Override
	public IArray getData() throws IOException {
		return mData;
	}

	@Override
	public IArray getData(int[] origin, int[] shape) throws IOException, InvalidRangeException {
		IArray result = getData();
		IArrayUtils util = result.getArrayUtils().section(origin, shape);
		result = util.getArray();
		return result;
	}

	@Override
	public String getDescription() {
		String result = "";
		IAttribute description = getAttribute("description");
		if( description != null && description.isString() ) {
			result = description.getStringValue();
		}
		return result;
	}

	@Override
	public List<IDimension> getDimensions(int index) {
		List<IDimension> result = new ArrayList<IDimension>();
		List<IDimension> dims = getParentGroup().getDimensionList();
		if( dims.size() > index ) {
			result.add( dims.get( index ) );
		}
		return result;
	}

	@Override
	public List<IDimension> getDimensionList() {
		return getParentGroup().getDimensionList();
	}

	@Override
	public String getDimensionsString() {
		StringBuffer result = new StringBuffer();
		int i = 0;
		for( IDimension dimension : getDimensionList() ) {
			if( i != 0 ) {
				result.append( " " );
			}
			result.append( dimension.getName() );
		}
		
		return result.toString();
	}

	@Override
	public int getElementSize() {
		throw new NotImplementedException();
	}

	@Override
	public String getNameAndDimensions() {
		StringBuffer result = new StringBuffer();
		getNameAndDimensions(result, false, false);
		return result.toString();
	}

	@Override
	public void getNameAndDimensions(StringBuffer buf, boolean longName, boolean length) {
		int i = 0;
		if( longName ) {
			buf.append( getName() );
		}
		else {
			buf.append( getShortName() );
		}
		for( IDimension dimension : getDimensionList() ) {
			if( i != 0 ) {
				buf.append( " " );
			}
			buf.append( dimension.getName() );
			if( length ) {
				buf.append( "(" );
				buf.append( dimension.getLength() );
				buf.append( ")" );
			}
		}
	}

	@Override
	public List<IRange> getRangeList() {
		throw new NotImplementedException();
	}

	@Override
	public int getRank() {
		return mData.getRank();
	}

	@Override
	public IDataItem getSection(List<IRange> section) throws InvalidRangeException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public List<IRange> getSectionRanges() {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public int[] getShape() {
		return mData.getShape();
	}

	@Override
	public long getSize() {
		return mData.getSize();
	}

	@Override
	public int getSizeToCache() {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public IDataItem getSlice(int dim, int value) throws InvalidRangeException {
		int[] shape = getShape();
		int[] origin = new int[getRank()];
		if( dim > getRank() - 1 && value > shape[dim] ) {
			throw new InvalidRangeException("Unable to create slice at the given position!");
		}
		shape[dim] = 1;
		origin[dim] = value;
		IArray array = null;
		IDataItem result = null;
		try {
			array = getData( origin, shape );
			result = clone();
			result.setCachedData(array, isMetadata() );
		} catch (IOException e) {
			throw new InvalidRangeException("Unable to create slice at the given position!");
		} catch (InvalidArrayTypeException e) {
			// Should not happen
		}

		return result;
	}

	@Override
	public Class<?> getType() {
		return mData.getElementType();
	}

	@Override
	public String getUnitsString() {
		String result = "";
		IAttribute attr = findAttributeIgnoreCase("unit");
		if( attr != null ) {
			result = attr.getStringValue();
		}
		return result;
	}

	@Override
	public boolean hasCachedData() {
		// Nothing to do because there is no cache
		return false;
	}

	@Override
	public void invalidateCache() {
		// Nothing to do because there is no cache
	}

	@Override
	public boolean isCaching() {
		// Nothing to do because there is no cache
		return false;
	}

	@Override
	public boolean isMemberOfStructure() {
		// Nothing to do no structure managed
		return false;
	}

	@Override
	public boolean isMetadata() {
		return false;
	}

	@Override
	public boolean isScalar() {
		return mData.getRank() == 0;
	}

	@Override
	public boolean isUnlimited() {
		return true;
	}

	@Override
	public boolean isUnsigned() {
		return false;
	}

	@Override
	public byte readScalarByte() throws IOException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public double readScalarDouble() throws IOException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public float readScalarFloat() throws IOException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public int readScalarInt() throws IOException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public long readScalarLong() throws IOException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public short readScalarShort() throws IOException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public String readScalarString() throws IOException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void setCachedData(IArray cacheData, boolean isMetadata) throws InvalidArrayTypeException {
		mData = cacheData;
	}

	@Override
	public void setCaching(boolean caching) {
		// Nothing to do: no cache
	}

	@Override
	public void setDataType(Class<?> dataType) {
		// TODO		
		throw new NotImplementedException();
	}

	@Override
	public void setDimensions(String dimString) {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void setDimension(IDimension dim, int ind) throws DimensionNotSupportedException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void setElementSize(int elementSize) {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void setSizeToCache(int sizeToCache) {
		// Nothing to do: no cache
	}

	@Override
	public void setUnitsString(String units) {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public String toStringDebug() {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public String writeCDL(String indent, boolean useFullName, boolean strict) {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void addOneAttribute(IAttribute attribute) {
		if( attribute != null ) {
			mAttributes.put(attribute.getName(), attribute);
		}
		
	}

	@Override
	public void addStringAttribute(String name, String value) {
		addOneAttribute( new ArchivingAttribute(mFactory, name, value) );
	}

	@Override
	public IAttribute getAttribute(String name) {
		return mAttributes.get(name);
	}

	@Override
	public List<IAttribute> getAttributeList() {
		return new ArrayList<IAttribute>( mAttributes.values() );
	}

	@Override
	public IDataset getDataset() {
		IDataset result = null;
		if( getParentGroup() != null ) {
			result = getParentGroup().getDataset();
		}
		return result;
	}

	@Override
	public String getLocation() {
		String result = "";
		if( getParentGroup() != null ) {
			result = getParentGroup().getDataset().getLocation();
		}
		return result;
	}

	@Override
	public String getName() {
		String result = "";
		if( mParent != null ) {
			result += mParent.getName();
		}
		result += AttributePath.SEPARATOR + getShortName();
		return result;
	}

	@Override
	public IGroup getParentGroup() {
		return mParent;
	}

	@Override
	public IGroup getRootGroup() {
		IGroup result = null;
		IDataset dataset = getDataset();
		if( dataset != null ) {
			result = getRootGroup();
		}
		return result;
	}

	@Override
	public String getShortName() {
		return mName;
	}

	@Override
	public boolean hasAttribute(String name, String value) {
		boolean result = false;
		if( name != null && value != null ) {
			IAttribute attr = mAttributes.get(name);
			if( 
				attr != null && attr.isString() && 
				! attr.isArray() && name.equals( attr.getName() ) && 
				value.equals( attr.getStringValue() ) 
			) {
				result = true;
			}
		}
		return result;
	}

	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		throw new NotImplementedException();
	}

	@Override
	public void setShortName(String name) {
		mName = name;
	}

	@Override
	public void setParent(IGroup group) {
		mParent = group;
	}

	@Override
	public long getLastModificationDate() {
		long result = 0;
		IDataset dataset = getDataset();
		if( dataset != null ) {
			result = dataset.getLastModificationDate();
		}
		return result;
	}

	@Override
	public String getFactoryName() {
		return mFactory;
	}

	@Override
	public boolean removeAttribute(IAttribute attribute) {
		boolean result = false;
		if( attribute != null ) {
			String name = attribute.getName();
			IAttribute attr = mAttributes.remove( name );
			result = ( attr != null );
		}
		return result;
	}
	
	@Override
	public String toString() {
		StringBuffer result = new StringBuffer();
		result.append( getName() );
		result.append("\nattrib: \n" );
		for( IAttribute attr : getAttributeList() ) {
			result.append("  - " + attr.toString() + "\n" );
		}
		return result.toString();
	}

}
