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
package org.cdma.plugin.soleil.nexus.navigation;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.hdf.navigation.HdfDataItem;
import org.cdma.engine.hdf.utils.HdfPath;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.plugin.soleil.nexus.NxsFactory;
import org.cdma.plugin.soleil.nexus.array.NxsArray;
import org.cdma.plugin.soleil.nexus.array.NxsIndex;
import org.cdma.plugin.soleil.nexus.utils.NxsPath;
import org.cdma.utils.Utilities.ModelType;

public final class NxsDataItem implements IDataItem, Cloneable {
    // Inner class
    // Associate a IDimension to an order of the array
    private static class DimOrder {
        // Members
        private final int mOrder; // order of the corresponding dimension in the NxsDataItem
        private final IDimension mDimension; // dimension object

        public DimOrder(int order, IDimension dim) {
            mOrder = order;
            mDimension = dim;
        }

        public int order() {
            return mOrder;
        }

        public IDimension dimension() {
            return mDimension;
        }
    }

    // / Members
    private NxsDataset mDataset; // CDMA IDataset i.e. file handler
    private IGroup mParent; // parent group
    private HdfDataItem[] mDataItems; // NeXus dataitem support of the data
    private NxsArray mArray; // CDMA IArray supporting a view of the data
    private final List<DimOrder> mDimension; // list of dimensions
    private NxsPath mPath;

    // / Constructors
    public NxsDataItem(String name, NxsDataset handler) {
        mDataset = handler;
        mDataItems = new HdfDataItem[] { new HdfDataItem(NxsFactory.NAME, name) };
        mDimension = new ArrayList<DimOrder>();
        mParent = null;
        mArray = null;
        mPath = null;
    }

    public NxsDataItem(final NxsDataItem dataItem) {
        mDataset = dataItem.mDataset;
        mDataItems = dataItem.mDataItems.clone();
        mDimension = new ArrayList<DimOrder>(dataItem.mDimension);
        mParent = dataItem.getParentGroup();
        mArray = null;
        mPath = dataItem.mPath;
        try {
            if (mDataItems.length == 1) {
                mArray = new NxsArray((NxsArray) dataItem.getData());
            } else {
                mArray = new NxsArray((NxsArray) dataItem.getData());
            }
        } catch (IOException e) {
        }
    }

    public NxsDataItem(HdfDataItem[] data, IGroup parent, NxsDataset handler) {
        mDataset = handler;
        if (data != null) {
            mDataItems = data.clone();
            mPath = new NxsPath(HdfPath.splitStringToNode(data[0].getLocation()));
        }
        mDimension = new ArrayList<DimOrder>();
        mParent = parent;
        mArray = null;
    }

    public NxsDataItem(HdfDataItem item, IGroup parent, NxsDataset dataset) {
        this(new HdfDataItem[] { item }, parent, dataset);
    }

    public NxsDataItem(NxsDataItem[] items, IGroup parent, NxsDataset dataset) {
        ArrayList<HdfDataItem> list = new ArrayList<HdfDataItem>();
        for (NxsDataItem cur : items) {
            for (HdfDataItem item : cur.mDataItems) {
                list.add(item);
            }
        }
        mPath = new NxsPath(HdfPath.splitStringToNode(items[0].getLocation()));
        mDataItems = list.toArray(new HdfDataItem[list.size()]);

        mDataset = dataset;
        mDimension = new ArrayList<DimOrder>();
        mParent = parent;
        mArray = null;
    }

    // / Methods
    @Override
    public ModelType getModelType() {
        return ModelType.DataItem;
    }

    @Override
    public List<IAttribute> getAttributeList() {
        return mDataItems[0].getAttributeList();
    }

    @Override
    public IArray getData() throws IOException {
        if (mArray == null && mDataItems.length > 0) {
            IArray[] arrays = new IArray[mDataItems.length];
            for (int i = 0; i < mDataItems.length; i++) {
                arrays[i] = mDataItems[i].getData();
            }
            mArray = new NxsArray(arrays);
        }
        return mArray;
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
        return new NxsDataItem(this);
    }

    @Override
    public void addOneAttribute(IAttribute att) {
        mDataItems[0].addOneAttribute(att);
    }

    @Override
    public void addStringAttribute(String name, String value) {
        mDataItems[0].addStringAttribute(name, value);
    }

    @Override
    public IAttribute getAttribute(String name) {
        IAttribute result = null;

        for (IDataItem item : mDataItems) {
            result = item.getAttribute(name);
            if (result != null) {
                break;
            }
        }

        return result;
    }

    @Override
    public IAttribute findAttributeIgnoreCase(String name) {
        IAttribute result = null;

        for (IDataItem item : mDataItems) {
            result = item.findAttributeIgnoreCase(name);
            if (result != null) {
                break;
            }
        }

        return result;
    }

    @Override
    public int findDimensionIndex(String name) {
        int result = -1;
        for (DimOrder dimord : mDimension) {
            if (dimord.mDimension.getName().equals(name)) {
                result = dimord.order();
                break;
            }
        }

        return result;
    }

    @Override
    public String getDescription() {
        String result = null;

        for (IDataItem item : mDataItems) {
            result = item.getDescription();
            if (result != null) {
                break;
            }
        }

        return result;
    }

    @Override
    public List<IDimension> getDimensions(int i) {
        ArrayList<IDimension> list = null;

        if (i <= getRank()) {
            list = new ArrayList<IDimension>();
            for (DimOrder dim : mDimension) {
                if (dim.order() == i) {
                    list.add(dim.dimension());
                }
            }
        }

        return list;
    }

    @Override
    public List<IDimension> getDimensionList() {
        ArrayList<IDimension> list = new ArrayList<IDimension>();

        for (DimOrder dimOrder : mDimension) {
            list.add(dimOrder.dimension());
        }

        return list;
    }

    @Override
    public String getDimensionsString() {
        StringBuffer dimList = new StringBuffer();

        int i = 0;
        for (DimOrder dim : mDimension) {
            if (i++ != 0) {
                dimList.append(" ");
            }
            dimList.append(dim.dimension().getName());
        }

        return dimList.toString();
    }

    @Override
    public int getElementSize() {
        return mDataItems[0].getElementSize();
    }

    @Override
    public String getName() {
        return mDataItems[0].getName();
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

        if (getRank() > 0) {
            buf.append("(");
        }
        for (int i = 0; i < mDimension.size(); i++) {
            DimOrder dim = mDimension.get(i);
            IDimension myd = dim.dimension();
            String dimName = myd.getName();
            if ((dimName == null) || !showDimLength) {
                dimName = "";
            }

            if (i != 0) {
                buf.append(", ");
            }

            if (myd.isVariableLength()) {
                buf.append("*");
            } else if (myd.isShared()) {
                if (!showDimLength) {
                    buf.append(dimName);
                    buf.append("=");
                    buf.append(myd.getLength());
                } else {
                    buf.append(dimName);
                }
            } else {
                if (dimName != null) {
                    buf.append(dimName);
                }
                buf.append(myd.getLength());
            }
        }
        if (getRank() > 0) {
            buf.append(")");
        }
    }

    @Override
    public IGroup getParentGroup() {
        return mParent;
    }

    @Override
    public List<IRange> getRangeList() {
        List<IRange> list = null;
        try {
            list = new NxsIndex(getData().getShape()).getRangeList();
        } catch (IOException e) {
        }
        return list;
    }

    @Override
    public List<IRange> getSectionRanges() {
        List<IRange> list = null;
        try {
            list = ((NxsIndex) getData().getIndex()).getRangeList();
        } catch (IOException e) {
        }
        return list;
    }

    @Override
    public int getRank() {
        int result;
        int[] shape = getShape();

        if (shape.length == 1 && shape[0] == 1) {
            result = 0;
        } else {
            result = shape.length;
        }

        return result;
    }

    @Override
    public IDataItem getSection(List<IRange> section) throws InvalidRangeException {
        NxsDataItem item = null;
        try {
            item = new NxsDataItem(this);
            mArray = (NxsArray) item.getData().getArrayUtils().sectionNoReduce(section).getArray();
        } catch (IOException e) {
        }
        return item;
    }

    @Override
    public int[] getShape() {
        int[] shape;
        if (mDataItems.length == 1) {
            shape = mDataItems[0].getShape();
        } else {
            try {
                shape = getData().getShape();
            } catch (IOException e) {
                shape = new int[] {};
            }
        }
        return shape;
    }

    @Override
    public String getShortName() {
        return mDataItems[0].getShortName();
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
        NxsDataItem item = new NxsDataItem(this);
        try {
            item.mArray = (NxsArray) item.getData().getArrayUtils().slice(dim, value).getArray();
        } catch (Exception e) {
            item = null;
        }
        return item;
    }

    @Override
    public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
        return getSlice(dimension, value);
    }

    @Override
    public Class<?> getType() {
        return mDataItems[0].getType();
    }

    @Override
    public String getUnitsString() {
        String value = null;
        IAttribute attr = getAttribute("unit");
        if (attr != null) {
            value = attr.getStringValue();
        }
        return value;
    }

    @Override
    public boolean hasAttribute(String name, String value) {
        boolean result = false;
        IAttribute attr = getAttribute(name);

        if (attr != null) {
            String test = attr.getStringValue();
            if (test.equals(value)) {
                result = true;
            }
        }

        return result;
    }

    @Override
    public boolean hasCachedData() {
        return mDataItems[0].hasCachedData();
    }

    @Override
    public void invalidateCache() {
        mDataItems[0].invalidateCache();
    }

    @Override
    public boolean isCaching() {
        return mDataItems[0].isCaching();
    }

    @Override
    public boolean isMemberOfStructure() {
        return mDataItems[0].isMemberOfStructure();
    }

    @Override
    public boolean isMetadata() {
        return (getAttribute("signal") == null);
    }

    @Override
    public boolean isScalar() {
        return (getRank() == 0);
    }

    @Override
    public boolean isUnlimited() {
        return false;
    }

    @Override
    public boolean isUnsigned() {
        return mDataItems[0].isUnsigned();
    }

    @Override
    public byte readScalarByte() throws IOException {
        return mDataItems[0].readScalarByte();
    }

    @Override
    public double readScalarDouble() throws IOException {
        return mDataItems[0].readScalarDouble();
    }

    @Override
    public float readScalarFloat() throws IOException {
        return mDataItems[0].readScalarFloat();
    }

    @Override
    public int readScalarInt() throws IOException {
        return mDataItems[0].readScalarInt();
    }

    @Override
    public long readScalarLong() throws IOException {
        return mDataItems[0].readScalarLong();
    }

    @Override
    public short readScalarShort() throws IOException {
        return mDataItems[0].readScalarShort();
    }

    @Override
    public String readScalarString() throws IOException {
        return mDataItems[0].readScalarString();
    }

    @Override
    public boolean removeAttribute(IAttribute attr) {
        boolean result = false;
        for (IDataItem item : mDataItems) {
            item.removeAttribute(attr);
        }
        result = true;
        return result;
    }

    @Override
    public void setCachedData(IArray cacheData, boolean isMetadata) throws InvalidArrayTypeException {
        if (cacheData instanceof NxsArray) {
            mArray = (NxsArray) cacheData;
            IArray[] parts = mArray.getArrayParts();
            for (int i = 0; i < parts.length && i < mDataItems.length; i++) {
                mDataItems[i].setCachedData(parts[i], false);
            }
        } else if (mDataItems.length == 1) {
            mDataItems[0].setCachedData(cacheData, isMetadata);
        } else {
            throw new InvalidArrayTypeException("Unable to set data: NxsArray is expected!");
        }
    }

    @Override
    public void setCaching(boolean caching) {
        for (IDataItem item : mDataItems) {
            item.setCaching(caching);
        }
    }

    @Override
    public void setDataType(Class<?> dataType) {
        for (IDataItem item : mDataItems) {
            item.setDataType(dataType);
        }
    }

    @Override
    public void setDimensions(String dimString) {
        mParent = getParentGroup();

        List<String> dimNames = java.util.Arrays.asList(dimString.split(" "));
        List<IDataItem> items = mParent.getDataItemList();

        for (IDataItem item : items) {
            IAttribute attr = item.getAttribute("axis");
            if (attr != null) {
                if ("*".equals(dimString)) {
                    setDimension(new NxsDimension(NxsFactory.NAME, item), attr.getNumericValue().intValue());
                } else if (dimNames.contains(attr.getName())) {
                    setDimension(new NxsDimension(NxsFactory.NAME, item), attr.getNumericValue().intValue());
                }
            }
        }
    }

    @Override
    public void setDimension(IDimension dim, int ind) {
        mDimension.add(new DimOrder(ind, dim));
    }

    @Override
    public void setElementSize(int elementSize) {
        for (IDataItem item : mDataItems) {
            item.setElementSize(elementSize);
        }
    }

    @Override
    public void setName(String name) {
        for (IDataItem item : mDataItems) {
            item.setName(name);
        }
    }

    @Override
    public void setParent(IGroup group) {
        if (mParent == null || !mParent.equals(group)) {
            mParent = group;
            for (HdfDataItem dataItem : mDataItems) {
                dataItem.setParent(((NxsGroup) group).getHdfGroup());
            }
            // TODO
            // group.addDataItem(this);
        }
    }

    @Override
    public void setSizeToCache(int sizeToCache) {
        for (IDataItem item : mDataItems) {
            item.setSizeToCache(sizeToCache);
        }
    }

    @Override
    public String toString() {
        return getName();
    }

    @Override
    public String toStringDebug() {
        StringBuffer strDebug = new StringBuffer();
        strDebug.append(getName());
        if (strDebug.length() > 0) {
            strDebug.append("\n");
        }
        try {
            strDebug.append("shape: " + getData().shapeToString() + "\n");
        } catch (IOException e) {
        }
        List<IDimension> dimensions = getDimensionList();
        for (IDimension dim : dimensions) {
            strDebug.append(dim.getCoordinateVariable().toString());
        }

        List<IAttribute> list = getAttributeList();
        if (list.size() > 0) {
            strDebug.append("\nAttributes:\n");
        }
        for (IAttribute a : list) {
            strDebug.append("- " + a.toString() + "\n");
        }

        return strDebug.toString();
    }

    @Override
    public String writeCDL(String indent, boolean useFullName, boolean strict) {
        throw new NotImplementedException();
    }

    @Override
    public void setUnitsString(String units) {
        throw new NotImplementedException();
    }

    @Override
    public IDataset getDataset() {
        return mDataset;
    }

    public void setDataset(IDataset dataset) {
        if (dataset instanceof NxsDataset) {
            mDataset = (NxsDataset) dataset;
        } else {
            try {
                mDataset = NxsDataset.instanciate(new File(dataset.getLocation()).toURI());
            } catch (NoResultException e) {
                Factory.getLogger().log(Level.WARNING, e.getMessage());
            }
        }
    }

    @Override
    public String getLocation() {
        String result = null;
        if (mDataItems != null && mDataItems.length > 0) {
            result = mDataItems[0].getLocation();
        }
        return result;
    }

    @Override
    public IGroup getRootGroup() {
        IGroup root;
        if (mParent != null) {
            root = mParent.getRootGroup();
        } else {
            root = mDataset.getRootGroup();
        }
        return root;
    }

    @Override
    public void setShortName(String name) {
        for (IDataItem item : mDataItems) {
            item.setShortName(name);
        }
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public long getLastModificationDate() {
        return mDataset.getLastModificationDate();
    }

    // specific methods
    public HdfDataItem[] getHdfDataItems() {
        HdfDataItem[] result = new HdfDataItem[mDataItems.length];
        int i = 0;
        for (HdfDataItem item : mDataItems) {
            result[i] = item;
            i++;
        }
        return result;
    }

    public NxsPath getPath() {
        return mPath;
    }

    public void setPath(NxsPath path) {
        mPath = path;
    }
}
