package org.cdma.plugin.archiving.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
import org.cdma.plugin.archiving.VcFactory;
import org.cdma.plugin.xml.navigation.XmlContainer;
import org.cdma.utils.IArrayUtils;
import org.cdma.utils.Utilities.ModelType;

public class VcDataItem extends XmlContainer implements IDataItem {

	private String mDataAttribute;
	private IArray mData;
	private final boolean attributeDataItem;

	// Constructor
	public VcDataItem(String name, IDataset dataset, IGroup parent, String dataAttribute) {
		super(VcFactory.NAME, name, VcGroup.INVALID_INDEX, dataset, parent);
		mDataAttribute = dataAttribute;
		mData = null;
		attributeDataItem = (isReadAttribute() || isWriteAttribute());
	}
	
	public VcDataItem(String name, IGroup parent, IArray data) {
		super(VcFactory.NAME, name, VcGroup.INVALID_INDEX, parent != null ? parent.getDataset() : null, parent);
		mData = data;
		attributeDataItem = true;
		mDataAttribute = name;
	}

	@Override
	public VcDataItem clone() {
		return new VcDataItem(getName(), getDataset(), getParentGroup(), mDataAttribute);
	}

	public void setAttributeData(String attributeName) {
		mDataAttribute = attributeName;
	}

	public boolean isReadAttribute() {
		return VcGroup.ATTRIBUTE_READ_DATA_LABEL.equalsIgnoreCase(getName());
	}

	public boolean isWriteAttribute() {
		return VcGroup.ATTRIBUTE_WRITE_DATA_LABEL.equalsIgnoreCase(getName());
	}

	@Override
	public ModelType getModelType() {
		return ModelType.DataItem;
	}

	public boolean isAttributeDataItem() {
		return attributeDataItem;
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
		String result = null;
		if (attributeDataItem) {
			result = mDataAttribute;
		} else {
			result = getShortName();
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
		return null;
	}

	@Override
	public List<IRange> getSectionRanges() {
		// TODO
		return null;
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
		return 0;
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
		// Nothing to do because there is no cache
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
		// TODO
		return false;
	}

	@Override
	public byte readScalarByte() throws IOException {
		// TODO
		return 0;
	}

	@Override
	public double readScalarDouble() throws IOException {
		// TODO
		return 0;
	}

	@Override
	public float readScalarFloat() throws IOException {
		// TODO
		return 0;
	}

	@Override
	public int readScalarInt() throws IOException {
		// TODO
		return 0;
	}

	@Override
	public long readScalarLong() throws IOException {
		// TODO
		return 0;
	}

	@Override
	public short readScalarShort() throws IOException {
		// TODO
		return 0;
	}

	@Override
	public String readScalarString() throws IOException {
		// TODO
		return null;
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
	}

	@Override
	public void setDimensions(String dimString) {
		// TODO
	}

	@Override
	public void setDimension(IDimension dim, int ind) throws DimensionNotSupportedException {
		// TODO
	}

	@Override
	public void setElementSize(int elementSize) {
		// TODO
	}

	@Override
	public void setSizeToCache(int sizeToCache) {
		// Nothing to do: no cache
	}

	@Override
	public void setUnitsString(String units) {
		// TODO
	}

	@Override
	public String toStringDebug() {
		// TODO
		return null;
	}

	@Override
	public String writeCDL(String indent, boolean useFullName, boolean strict) {
		// TODO
		return null;
	}

	public void printHierarchy(int level) {

		String tabFormatting = "";
		for (int i = 0; i < level; ++i) {
			tabFormatting += "\t";
		}
		// Tag beginning
		System.out.print(tabFormatting);
		System.out.print("<");
		System.out.print(getShortName());

		// Attributes of this group
		for (IAttribute attr : getAttributeList()) {
			System.out.print(" ");
			System.out.print(attr.getName());
			System.out.print("=\"");
			System.out.print(attr.getStringValue());
			System.out.print("\"");
		}
		// Tag Ending
		System.out.print("/>\n");
	}

}
