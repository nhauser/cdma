package org.cdma.plugin.archiving.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.cdma.exception.DimensionNotSupportedException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IRange;
import org.cdma.plugin.archiving.VcFactory;
import org.cdma.plugin.archiving.array.VcArray;
import org.cdma.plugin.xml.navigation.XmlContainer;
import org.cdma.utils.Utilities.ModelType;

import fr.soleil.commonarchivingapi.ArchivingTools.Tools.DbData;
import fr.soleil.commonarchivingapi.ArchivingTools.Tools.NullableTimedData;

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

/*	
	protected void initArrayFromDatabase(DbData data) {
		if (data != null) {
			NullableTimedData[] subDatas = data.getData_timed();
			if (subDatas != null) {
				int[] shape;
				Object storage = null;
				if (data.getMax_x() > 1) {
					// if max_x > 1, the data is a spectrum
					shape = new int[3];
					shape[0] = data.size();
					shape[1] = 2;
					shape[2] = data.getMax_x();

					Object[][] tempStorage = new Object[shape[0]][shape[1]];
					for (int i = 0; i < shape[0]; ++i) {
						Object valueSample = subDatas[i].value;
						if (valueSample != null
								&& valueSample.getClass().isArray()) {
							Object[] arraySample = (Object[]) valueSample;
							if (arraySample.length > 0) {
								tempStorage[i][VcArray.VALUE_INDEX] = Arrays
										.copyOf(arraySample, arraySample.length);
								tempStorage[i][VcArray.TIME_INDEX] = subDatas[i].time;
							}
						}
					}
					storage = tempStorage;
				} else {
					// else it is a scalar (image are not handle)
					shape = new int[2];
					shape[0] = data.size();
					shape[1] = 2;

					Object[] tempStorage = new Object[shape[0] * shape[1]];

					for (int i = 0; i < shape[0]; ++i) {
						tempStorage[i * 2] = subDatas[i].time;
						Object valueSample = subDatas[i].value;
						if (valueSample != null
								&& valueSample.getClass().isArray()) {
							Object[] arraySample = (Object[]) valueSample;
							if (arraySample.length == 1) {
								tempStorage[i * 2 + 1] = arraySample[0];
							}
						}
					}
					storage = tempStorage;
				}
				mData = new VcArray(storage, shape);
			}
		}
	}
	 */
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
		return 0;
	}

	@Override
	public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
		return null;
	}

	@Override
	public IArray getData() throws IOException {
		return mData;
	}

	@Override
	public IArray getData(int[] origin, int[] shape) throws IOException, InvalidRangeException {
		return null;
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
		return null;
	}

	@Override
	public int getElementSize() {
		return -1;
	}

	@Override
	public String getNameAndDimensions() {
		return null;
	}

	@Override
	public void getNameAndDimensions(StringBuffer buf, boolean longName, boolean length) {
	}

	@Override
	public List<IRange> getRangeList() {
		return null;
	}

	@Override
	public int getRank() {
		return mData.getRank();
	}

	@Override
	public IDataItem getSection(List<IRange> section) throws InvalidRangeException {
		return null;
	}

	@Override
	public List<IRange> getSectionRanges() {
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
		return 0;
	}

	@Override
	public IDataItem getSlice(int dim, int value) throws InvalidRangeException {
		return null;
	}

	@Override
	public Class<?> getType() {
		return mData.getElementType();
	}

	@Override
	public String getUnitsString() {
		return null;
	}

	@Override
	public boolean hasCachedData() {
		return false;
	}

	@Override
	public void invalidateCache() {
	}

	@Override
	public boolean isCaching() {
		return false;
	}

	@Override
	public boolean isMemberOfStructure() {
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
		return false;
	}

	@Override
	public boolean isUnsigned() {
		return false;
	}

	@Override
	public byte readScalarByte() throws IOException {
		return 0;
	}

	@Override
	public double readScalarDouble() throws IOException {
		return 0;
	}

	@Override
	public float readScalarFloat() throws IOException {
		return 0;
	}

	@Override
	public int readScalarInt() throws IOException {
		return 0;
	}

	@Override
	public long readScalarLong() throws IOException {
		return 0;
	}

	@Override
	public short readScalarShort() throws IOException {
		return 0;
	}

	@Override
	public String readScalarString() throws IOException {
		return null;
	}

	@Override
	public void setCachedData(IArray cacheData, boolean isMetadata) throws InvalidArrayTypeException {
	}

	@Override
	public void setCaching(boolean caching) {
	}

	@Override
	public void setDataType(Class<?> dataType) {
	}

	@Override
	public void setDimensions(String dimString) {
	}

	@Override
	public void setDimension(IDimension dim, int ind) throws DimensionNotSupportedException {
	}

	@Override
	public void setElementSize(int elementSize) {
	}

	@Override
	public void setSizeToCache(int sizeToCache) {
	}

	@Override
	public void setUnitsString(String units) {
	}

	@Override
	public String toStringDebug() {
		return null;
	}

	@Override
	public String writeCDL(String indent, boolean useFullName, boolean strict) {
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
