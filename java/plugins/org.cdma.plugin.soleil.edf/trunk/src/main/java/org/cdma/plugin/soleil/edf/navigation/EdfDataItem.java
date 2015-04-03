/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.edf.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.cdma.arrays.DefaultCompositeArray;
import org.cdma.arrays.DefaultRange;
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
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.plugin.soleil.edf.EdfFactory;
import org.cdma.plugin.soleil.edf.abstraction.AbstractObject;
import org.cdma.plugin.soleil.edf.array.BasicDimension;
import org.cdma.plugin.soleil.edf.utils.StringUtils;
import org.cdma.utils.Utilities.ModelType;

public class EdfDataItem extends AbstractObject implements IDataItem {

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
    private IIndex orignalShape;
    private EdfDataItem[] dataItems;
    protected IArray data;

    private EdfDataItem(EdfDataItem item) {

        this.data = item.getData();

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
        dataItems = new EdfDataItem[1];
        dataItems[0] = this;
    }

    public EdfDataItem(String name, IArray value) {
        this(name, value, false);
    }

    public EdfDataItem(EdfDataItem[] items) {
        super();
        ArrayList<EdfDataItem> list = new ArrayList<EdfDataItem>();
        for (EdfDataItem cur : items) {
            list.add(cur);
        }
        dataItems = list.toArray(new EdfDataItem[list.size()]);
        this.dimensions = new ArrayList<DimOrder>();
        if (data == null && dataItems.length > 0) {
            IArray[] arrays = new IArray[dataItems.length];
            for (int i = 0; i < dataItems.length; i++) {
                arrays[i] = dataItems[i].getData();
            }
            data = new DefaultCompositeArray(EdfFactory.NAME, arrays);
        }
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DataItem;
    }

    @Override
    public boolean hasAttribute(String name, String value) {
        boolean result = super.hasAttribute(name, value);
        if (!result && data instanceof DefaultCompositeArray && dataItems != null && dataItems.length > 0) {
            result = dataItems[0].hasAttribute(name, value);
        }
        return result;
    }

    @Override
    public IAttribute findAttributeIgnoreCase(String name) {
        if (attributes != null) {
            for (IAttribute attribute : attributes) {
                if (StringUtils.isSameStringIgnoreCase(name, attribute.getName())) {
                    return attribute;
                }
            }
        }
        return null;
    }

    @Override
    public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
        return getSlice(dimension, value);
    }

    @Override
    public IGroup getRootGroup() {
        IGroup result = null;
        IDataItem dataItem = dataItems[0];
        if (dataItem != null) {
            IContainer parent = dataItem.getParentGroup();
            if (parent != null) {
                result = (IGroup) parent.getRootGroup();
            }
            else {
                result = (IGroup) parent;
            }
        }
        return result;
    }

    @Override
    public IArray getData() {
        return data;
    }

    @Override
    public IArray getData(int[] origin, int[] shape) throws IOException, InvalidRangeException {
        IArray array = getData().copy(false);
        IIndex index = array.getIndex();

        if (shape == null || shape.length != array.getRank()) {
            throw new InvalidRangeException("Shape must be of same rank as the array!");
        }
        if (origin == null || origin.length != array.getRank()) {
            throw new InvalidRangeException("Origin must be of same rank as the array!");
        }

        int str = 1;
        long[] stride = new long[array.getRank()];
        for (int i = array.getRank() - 1; i >= 0; i--) {
            stride[i] = str;
            str *= shape[i];
        }
        index.setStride(stride);
        index.setShape(shape);
        index.setOrigin(origin);
        array.setIndex(index);
        return array;
    }

    @Override
    public IDataItem clone() {
        return (IDataItem) super.clone();
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
            try {
                list.add(new DefaultRange("", origin[i], shape[i] * stride[i], stride[i]));
            }
            catch (InvalidRangeException e) {

                e.printStackTrace();
            }
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
        throw new NotImplementedException();
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

        item = new EdfDataItem(this);
        item.data.setIndex(item.getData().getArrayUtils().sectionNoReduce(section).getArray()
                .getIndex());
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
            try {
                list.add(new DefaultRange("", origin[i], shape[i] * stride[i], stride[i]));
            }
            catch (InvalidRangeException e) {
                e.printStackTrace();
            }
        }

        return list;
    }

    @Override
    public int[] getShape() {
        int[] shape;
        //        if (dataItems.length == 1) {
        //            shape = dataItems[0].getShape();
        //        }
        //        else {
        shape = getData().getShape();
        //        }
        return shape;
    }

    @Override
    public long getSize() {
        int[] shape = getShape();
        long total = 1;
        for (int size : shape) {
            total *= size;
        }

        return total;
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
        return data.getElementType();
    }

    @Override
    public String getUnitsString() {
        return getAttribute("units").getStringValue();
    }

    @Override
    public boolean hasCachedData() {
        return dataItems[0].hasCachedData();
    }

    @Override
    public void invalidateCache() {
        dataItems[0].invalidateCache();

    }

    @Override
    public boolean isCaching() {
        return dataItems[0].isCaching();
    }

    @Override
    public boolean isMemberOfStructure() {
        return dataItems[0].isMemberOfStructure();
    }

    @Override
    public boolean isMetadata() {
        throw new NotImplementedException();
    }

    @Override
    public boolean isScalar() {
        return (data.getRank() == 0 || (data.getRank() == 1 && data.getShape()[0] == 1));
    }

    @Override
    public byte readScalarByte() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public double readScalarDouble() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public float readScalarFloat() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public int readScalarInt() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public long readScalarLong() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public short readScalarShort() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public String readScalarString() throws IOException {
        String result = null;
        if (data != null) {
            String[] stringArray = (String[]) data.getArrayUtils().copyTo1DJavaArray();
            result = stringArray[0];
        }
        return result;
    }

    @Override
    public void setCachedData(IArray cacheData, boolean isMetadata)
            throws InvalidArrayTypeException {
        data = cacheData;
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
        IGroup parent = getParentGroup();

        List<String> dimNames = java.util.Arrays.asList(dimString.split(" "));
        List<IDataItem> items = parent.getDataItemList();

        for (IDataItem item : items) {
            IAttribute attr = item.getAttribute("axis");
            if (attr != null) {
                try {
                    IDimension dim = new BasicDimension(item.getName(), item.getData(), false,
                            item.isUnlimited(), false);
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
        return getRootGroup().getLastModificationDate();
    }

    @Override
    public String getFactoryName() {
        return EdfFactory.NAME;
    }

    @Override
    public String toString() {
        StringBuffer buffer = new StringBuffer();
        buffer.append("DataItem " + getName());
        return buffer.toString();
    }
}
