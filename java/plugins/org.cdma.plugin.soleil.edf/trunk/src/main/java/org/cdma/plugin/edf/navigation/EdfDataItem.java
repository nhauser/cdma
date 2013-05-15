package org.cdma.plugin.edf.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.plugin.edf.abstraction.AbstractDataItem;
import org.cdma.plugin.edf.array.BasicDimension;
import org.cdma.plugin.edf.array.BasicRange;

public class EdfDataItem extends AbstractDataItem {
    // Inner class
    // Associate a IDimension to an order of the array
    private class DimOrder {
        // Members
        public int order; // order of the corresponding dimension in the NxsDataItem
        public IDimension dimension; // dimension object

        public DimOrder(int ord, IDimension dim) {
            order = ord;
            dimension = dim;
        }
    }

    // Members
    private boolean unsigned = false; // is the worn array unsigned or not
    private final ArrayList<DimOrder> dimensions; // list of dimensions
    private final IIndex orignalShape;

    private EdfDataItem(EdfDataItem item) {
        try {
            this.data = item.getData();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        this.dimensions = item.dimensions;
        this.unsigned = item.unsigned;
        this.parentGroup = item.parentGroup;
        this.orignalShape = item.orignalShape;
    }

    public EdfDataItem(String name, IArray value, boolean unsigned) {
        super();
        setName(name);
        this.data = value;
        this.unsigned = unsigned;
        this.orignalShape = value.getIndex();
        this.dimensions = new ArrayList<EdfDataItem.DimOrder>();
    }

    public EdfDataItem(String name, IArray value) {
        super();
        setName(name);
        this.data = value;
        this.orignalShape = value.getIndex();
        this.dimensions = new ArrayList<EdfDataItem.DimOrder>();
    }

    @Override
    public String getDescription() {
        IAttribute attr = getAttribute("long_name");
        if (attr == null) {
            attr = getAttribute("description");
        }
        if (attr == null) {
            attr = getAttribute("title");
        }
        if (attr == null) {
            attr = getAttribute("standard_name");
        }
        return attr.getStringValue();
    }

    @Override
    public List<IRange> getRangeList() {
        int rank = data.getRank();

        List<IRange> list = new ArrayList<IRange>();
        IIndex idx = data.getIndex();
        int[] origin = idx.getOrigin();
        int[] shape = idx.getShape();
        long[] stride = idx.getStride();

        for (int i = 0; i < rank; i++) {
            list.add(new BasicRange("", origin[i], shape[i] * stride[i], stride[i]));
        }

        return list;
    }

    @Override
    public int getRank() {
        return data.getRank();
    }

    @Override
    public boolean isUnlimited() {
        return false;
    }

    @Override
    public boolean isUnsigned() {
        return unsigned;
    }

    @Override
    public List<IDimension> getDimensions(int i) {
        ArrayList<IDimension> list = new ArrayList<IDimension>();

        for (DimOrder dim : dimensions) {
            if (dim.order == i) {
                list.add(dimensions.get(i).dimension);
            }
        }

        if (list.size() > 0)
            return list;
        else
            return null;
    }

    @Override
    public List<IDimension> getDimensionList() {
        ArrayList<IDimension> list = new ArrayList<IDimension>();

        for (DimOrder dimOrder : dimensions) {
            list.add(dimOrder.dimension);
        }

        return list;
    }

    @Override
    public String getDimensionsString() {
        String dimList = "";

        int i = 0;
        for (DimOrder dim : dimensions) {
            if (i++ != 0) {
                dimList += " ";
            }
            dimList += dim.dimension.getName();
        }

        return dimList;
    }

    @Override
    public int findDimensionIndex(String name) {
        for (DimOrder dimord : dimensions) {
            if (dimord.dimension.getName().equals(name))
                return dimord.order;
        }

        return -1;
    }

    @Override
    public int getElementSize() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public String getNameAndDimensions() {
        StringBuffer buf = new StringBuffer();
        getNameAndDimensions(buf, true, false);
        return buf.toString();
    }

    @Override
    public void getNameAndDimensions(StringBuffer buf, boolean useFullName, boolean showDimLength) {
        useFullName = useFullName && !showDimLength;
        String name = useFullName ? getName() : getShortName();
        buf.append(name);

        if (getRank() > 0)
            buf.append("(");
        for (int i = 0; i < dimensions.size(); i++) {
            DimOrder dim = dimensions.get(i);
            IDimension myd = dim.dimension;
            String dimName = myd.getName();
            if ((dimName == null) || !showDimLength)
                dimName = "";

            if (i != 0)
                buf.append(", ");

            if (myd.isVariableLength()) {
                buf.append("*");
            }
            else if (myd.isShared()) {
                if (!showDimLength)
                    buf.append(dimName + "=" + myd.getLength());
                else
                    buf.append(dimName);
            }
            else {
                if (dimName != null) {
                    buf.append(dimName);
                }
                buf.append(myd.getLength());
            }
        }

        if (getRank() > 0)
            buf.append(")");
    }

    @Override
    public IDataItem getSection(List<IRange> section) throws InvalidRangeException {
        EdfDataItem item = null;
        try {
            item = new EdfDataItem(this);
            item.data.setIndex(item.getData().getArrayUtils().sectionNoReduce(section).getArray()
                    .getIndex());
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return item;
    }

    @Override
    public List<IRange> getSectionRanges() {
        int rank = orignalShape.getRank();

        List<IRange> list = new ArrayList<IRange>();
        int[] origin = orignalShape.getOrigin();
        int[] shape = orignalShape.getShape();
        long[] stride = orignalShape.getStride();

        for (int i = 0; i < rank; i++) {
            list.add(new BasicRange("", origin[i], shape[i] * stride[i], stride[i]));
        }

        return list;
    }

    @Override
    public int[] getShape() {
        return data.getShape();
    }

    @Override
    public long getSize() {
        return data.getSize();
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
        return data.getElementType();
    }

    @Override
    public String getUnitsString() {
        return getAttribute("units").getStringValue();
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
        return (data.getRank() == 0 || (data.getRank() == 1 && data.getShape()[0] == 1));
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
    public void setCachedData(IArray cacheData, boolean isMetadata)
            throws InvalidArrayTypeException {
        data = cacheData;
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
        IGroup parent = getParentGroup();

        List<String> dimNames = java.util.Arrays.asList(dimString.split(" "));
        List<IDataItem> items = parent.getDataItemList();

        for (IDataItem item : items) {
            IAttribute attr = item.getAttribute("axis");
            if (attr != null) {
                try {
                    IDimension dim = new BasicDimension(item.getName(), item.getData(), false, item
                            .isUnlimited(), false);
                    if ("*".equals(dimString)) {
                        setDimension(dim, attr.getNumericValue().intValue());
                    }
                    else if (dimNames.contains(attr.getName())) {
                        setDimension(dim, attr.getNumericValue().intValue());
                    }
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    @Override
    public void setDimension(IDimension dim, int ind) {
        dimensions.add(new DimOrder(ind, dim));
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
    public void addStringAttribute(String name, String value) {
        addOneAttribute(new EdfAttribute(name, value));
    }

    @Override
    public IDataset getDataset() {
        return getRootGroup().getDataset();
    }

    @Override
    public String getLocation() {
        return getParentGroup().getLocation() + "/" + getName();
    }

    @Override
    public String getShortName() {
        return getName();
    }

    @Override
    public void setShortName(String name) {
        this.name = name;

    }

    @Override
    public long getLastModificationDate() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public String getFactoryName() {
        // TODO Auto-generated method stub
        return null;
    }
}
