/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.hdf.navigation;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;

import ncsa.hdf.hdf5lib.HDF5Constants;
import ncsa.hdf.hdf5lib.exceptions.HDF5Exception;
import ncsa.hdf.hdf5lib.exceptions.HDF5LibraryException;
import ncsa.hdf.object.Attribute;
import ncsa.hdf.object.Datatype;
import ncsa.hdf.object.FileFormat;
import ncsa.hdf.object.Group;
import ncsa.hdf.object.HObject;
import ncsa.hdf.object.h5.H5Datatype;
import ncsa.hdf.object.h5.H5File;
import ncsa.hdf.object.h5.H5ScalarDS;

import org.cdma.Factory;
import org.cdma.engine.hdf.array.HdfArray;
import org.cdma.engine.hdf.array.HdfIndex;
import org.cdma.engine.hdf.utils.HdfObjectUtils;
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
import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.utils.DataType;
import org.cdma.utils.Utilities.ModelType;

public class HdfDataItem implements IDataItem, Cloneable {

    private H5ScalarDS h5Item;
    protected H5File h5File;
    protected final String factoryName;
    protected IArray array;
    protected IGroup parent;
    protected String shortName;
    protected final List<IAttribute> attributeList = new ArrayList<IAttribute>();
    protected boolean dirty = false;
    private boolean isLink = false;

    public HdfDataItem(final String factoryName, final H5File file, final IGroup parent, final H5ScalarDS dataset) {
        this.factoryName = factoryName;
        this.h5File = file;
        this.h5Item = dataset;
        this.parent = parent;
        if (this.h5Item != null) {
            this.h5Item.init();
            this.shortName = h5Item.getName();
            loadAttributes();
        } else {
            dirty = true;
            this.shortName = "";
        }
    }

    public HdfDataItem(final String factoryName, final String name) {
        this(factoryName, null, null, null);
        this.shortName = name;
    }

    private HdfDataItem(final HdfDataItem dataItem) {
        this.parent = dataItem.parent;
        this.factoryName = dataItem.getFactoryName();
        this.h5File = dataItem.getH5File();
        this.h5Item = dataItem.getH5DataItem();
        this.shortName = dataItem.shortName;
        // this.fullName = dataItem.fullName;
        if (this.h5Item != null) {
            this.h5Item.init();
        }
        try {
            this.array = dataItem.getData();
            dirty = true;
        } catch (IOException e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public IDataItem clone() {
        return new HdfDataItem(this);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DataItem;
    }

    public H5ScalarDS getH5DataItem() {
        return this.h5Item;
    }

    public H5File getH5File() {
        return this.h5File;
    }

    public void linkTo(HdfDataItem dataitem) {
        if (dataitem != null) {
            this.isLink = true;
            this.h5Item = dataitem.getH5DataItem();
        } else {
            this.isLink = false;
        }
    }

    @Override
    public void addOneAttribute(final IAttribute attribute) {
        this.attributeList.add(attribute);
    }

    @Override
    public void addStringAttribute(final String name, final String value) {
        IAttribute attr = new HdfAttribute(factoryName, name, value);
        addOneAttribute(attr);
    }

    public IAttribute getAttribute(final String name, final boolean ignoreCase) {
        IAttribute result = null;
        for (IAttribute attribute : attributeList) {
            if ((ignoreCase && name.equalsIgnoreCase(attribute.getName())) || name.equals(attribute.getName())) {
                result = attribute;
            }
        }
        return result;
    }

    protected void loadAttributes() {
        List<?> attributes;
        try {
            if (h5Item != null) {
                attributes = h5Item.getMetadata();
                for (Object attribute : attributes) {
                    IAttribute attr = new HdfAttribute(factoryName, (Attribute) attribute);
                    this.attributeList.add(attr);
                }
            }
        } catch (HDF5Exception e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to loadAttributes for " + this.getName(), e);
        }
    }

    @Override
    public IAttribute getAttribute(final String name) {
        IAttribute result = null;
        result = getAttribute(name, false);
        return result;
    }

    @Override
    public List<IAttribute> getAttributeList() {
        return attributeList;
    }

    @Override
    public IDataset getDataset() {
        IDataset result = null;
        IContainer parentGroup = getParentGroup();
        if (parentGroup != null) {
            result = parentGroup.getDataset();
        }
        return result;
    }

    @Override
    public String getLocation() {
        String result = null;
        IContainer parentGroup = getParentGroup();
        if (parentGroup != null) {
            result = parentGroup.getLocation();
            // result = result + "/" + getName();
        }
        return result;
    }

    @Override
    public String getName() {
        IContainer parent = getParentGroup();
        return (parent == null ? "" : parent.getName() + "/") + getShortName();
        // return getShortName();
    }

    @Override
    public IContainer getParentGroup() {
        return parent;
    }

    @Override
    public IContainer getRootGroup() {
        IContainer result = null;
        IContainer parent = getParentGroup();
        if (parent != null) {
            result = parent.getRootGroup();
        }
        return result;
    }

    @Override
    public String getShortName() {
        return shortName;
    }

    @Override
    public boolean hasAttribute(final String name, final String value) {
        boolean result = HdfObjectUtils.hasAttribute(h5Item, name, value);
        return result;
    }

    @Override
    public void setName(final String name) {
        String[] nodes = name.split("/");
        int depth = nodes.length - 1;

        if (depth >= 0) {
            setShortName(nodes[depth--]);
        }

        IGroup group = (IGroup) getParentGroup();
        while (group != null && !group.isRoot() && depth >= 0) {
            group.setShortName(nodes[depth--]);
        }
    }

    @Override
    public void setShortName(final String name) {
        this.shortName = name;
    }

    @Override
    public void setParent(final IGroup group) {
        try {
            parent = group;
            if (h5Item != null) {
                h5Item.setPath(group.getName());
            }
        } catch (Exception e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to setParent", e);
        }
    }

    @Override
    public long getLastModificationDate() {
        long result = 0;
        if (h5Item != null) {
            String fileName = h5Item.getFile();
            File currentFile = new File(fileName);
            if (currentFile != null && currentFile.exists()) {
                result = currentFile.lastModified();
            }
        }
        return result;
    }

    @Override
    public String getFactoryName() {
        return this.factoryName;
    }

    @Override
    public IAttribute findAttributeIgnoreCase(final String name) {
        IAttribute result = null;
        result = getAttribute(name, true);
        return result;
    }

    @Override
    public int findDimensionIndex(final String name) {
        return 0;
    }

    @Override
    public IDataItem getASlice(final int dimension, final int value) throws InvalidRangeException {
        return getSlice(dimension, value);
    }

    @Override
    public IArray getData() throws IOException {
        if (array == null) {
            int[] shape = getShape();
            int[] origin = new int[getRank()];
            try {
                array = getData(origin, shape);
            } catch (Exception e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to initialize data!", e);
            }
        }
        return array;
    }

    @Override
    public IArray getData(final int[] origin, final int[] shape) throws IOException, InvalidRangeException {
        IArray array = null;
        IIndex index = null;
        try {
            Class<?> type = getType();
            if (type != null) {
                array = new HdfArray(factoryName, getType(), new int[0], this);
                index = new HdfIndex(this.factoryName, shape, origin, shape);
                array.setIndex(index);
            }
        } catch (InvalidArrayTypeException e) {
            throw new InvalidRangeException(e);
        }

        return array;
    }

    public Object load(final int[] origin, final int[] shape) throws OutOfMemoryError, Exception {
        Object result = null;
        Object tempResult = null;
        boolean viewHasChanged = false;
        synchronized (h5Item) {

            if (origin.length == h5Item.getDims().length || shape.length == origin.length) {
                // Set Selected dimensions
                long[] sDims = h5Item.getSelectedDims();

                for (int i = 0; i < sDims.length; i++) {
                    sDims[i] = 1;
                }

                if (!Arrays.equals(shape, HdfObjectUtils.convertLongToInt(sDims))) {
                    viewHasChanged = true;
                    for (int i = 0; i < shape.length; i++) {
                        sDims[i] = shape[i];
                    }
                }

                // Set origin
                long[] startDims = h5Item.getStartDims();
                if (!Arrays.equals(origin, HdfObjectUtils.convertLongToInt(startDims))) {
                    viewHasChanged = true;
                    for (int i = 0; i < origin.length; i++) {
                        startDims[i] = origin[i];
                    }
                }

                if (viewHasChanged) {
                    h5Item.clear();
                }
                tempResult = h5Item.getData();
                h5Item.clear();
                result = tempResult;
            }
        }
        return result;
    }

    @Override
    public String getDescription() {
        String result = null;
        IAttribute attribute = null;

        result = getAttribute("long_name").getStringValue();
        if (attribute == null) {
            result = getAttribute("description").getStringValue();
        }
        if (attribute == null) {
            result = getAttribute("title").getStringValue();
        }
        if (attribute == null) {
            result = getAttribute("standard_name").getStringValue();
        }
        if (attribute == null) {
            result = getAttribute("name").getStringValue();
        }
        return result;
    }

    @Override
    public List<IDimension> getDimensions(final int index) {
        return new ArrayList<IDimension>();
    }

    @Override
    public List<IDimension> getDimensionList() {
        return new ArrayList<IDimension>();
    }

    @Override
    public String getDimensionsString() {
        return "";
    }

    @Override
    public int getElementSize() {
        int result = 0;
        if (h5Item != null) {
            Datatype type = h5Item.getDatatype();
            if (type != null) {
                result = type.getDatatypeSize();
            }
        }
        return result;
    }

    @Override
    public String getNameAndDimensions() {
        return null;
    }

    @Override
    public void getNameAndDimensions(final StringBuffer buf, final boolean longName, final boolean length) {
    }

    @Override
    public List<IRange> getRangeList() {
        List<IRange> list = new ArrayList<IRange>();
        try {
            HdfIndex index = (HdfIndex) getData().getIndex();
            list.addAll(index.getRangeList());
        } catch (IOException e) {
            list = null;
        }
        return list;
    }

    @Override
    public int getRank() {
        int result = 0;
        if (h5Item != null) {
            result = h5Item.getRank();
        }
        return result;
    }

    @Override
    public IDataItem getSection(final List<IRange> section) throws InvalidRangeException {
        HdfDataItem item = null;
        try {
            item = new HdfDataItem(this);
            item.array = item.getData().getArrayUtils().sectionNoReduce(section).getArray();
        } catch (IOException e) {
        }
        return item;
    }

    @Override
    public List<IRange> getSectionRanges() {
        List<IRange> list = new ArrayList<IRange>();
        try {
            HdfIndex index = (HdfIndex) getData().getIndex();
            list.addAll(index.getRangeList());
        } catch (IOException e) {
            list = null;
        }
        return list;
    }

    @Override
    public int[] getShape() {
        int[] result = null;
        if (array != null) {
            result = array.getShape();
        } else {
            if (h5Item != null) {
                long[] dims = h5Item.getDims();
                result = HdfObjectUtils.convertLongToInt(dims);
            }
        }
        return result;
    }

    @Override
    public long getSize() {
        long size = 0;
        if (h5Item != null) {
            long[] dims = h5Item.getDims();
            for (int i = 0; i < dims.length; i++) {
                if (dims[i] >= 0)
                    size *= dims[i];
            }
        }
        return size;
    }

    @Override
    public int getSizeToCache() {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem getSlice(final int dim, final int value) throws InvalidRangeException {
        HdfDataItem item = new HdfDataItem(this);
        try {
            item.array = item.getData().getArrayUtils().slice(dim, value).getArray();
        } catch (Exception e) {
            item = null;
        }
        return item;
    }

    @Override
    public Class<?> getType() {
        Class<?> result = null;
        if (h5Item != null) {
            Datatype dType = h5Item.getDatatype();

            if (dType != null) {
                int datatype = dType.getDatatypeClass();

                if (DataType.BOOLEAN.equals(datatype)) {
                    result = Boolean.TYPE;
                } else if (Datatype.CLASS_CHAR == datatype) {
                    result = Byte.TYPE;
                } else if (Datatype.CLASS_FLOAT == datatype) {
                    switch (dType.getDatatypeSize()) {
                        case 4:
                            result = Float.TYPE;
                            break;
                        case 8:
                            result = Double.TYPE;
                            break;
                        default:
                            break;
                    }
                } else if (Datatype.CLASS_INTEGER == datatype) {
                    switch (dType.getDatatypeSize()) {
                        case 1:
                            result = Byte.TYPE;
                            break;
                        case 2:
                            result = Short.TYPE;
                            break;
                        case 4:
                            result = Integer.TYPE;
                            break;
                        case 8:
                            result = Long.TYPE;
                            break;
                        default:
                            break;
                    }

                } else if (Datatype.CLASS_STRING == datatype) {
                    result = String.class;
                }
            }
        } else if (array != null) {
            result = array.getElementType();
        }
        return result;
    }

    @Override
    public String getUnitsString() {
        IAttribute attr = getAttribute("unit");
        String value = null;
        if (attr != null) {
            value = attr.getStringValue();
        }
        return value;
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
        boolean result = false;
        if (h5Item != null) {
            result = h5Item.getDatatype().isUnsigned();
        }
        return result;
    }

    @Override
    public byte readScalarByte() throws IOException {
        try {
            return java.lang.reflect.Array.getByte(h5Item.getData(), 0);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    @Override
    public double readScalarDouble() throws IOException {
        try {
            return java.lang.reflect.Array.getDouble(h5Item.getData(), 0);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public float readScalarFloat() throws IOException {
        try {
            return java.lang.reflect.Array.getFloat(h5Item.getData(), 0);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int readScalarInt() throws IOException {
        try {
            return java.lang.reflect.Array.getInt(h5Item.getData(), 0);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public long readScalarLong() throws IOException {
        try {
            return java.lang.reflect.Array.getLong(h5Item.getData(), 0);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public short readScalarShort() throws IOException {
        try {
            return java.lang.reflect.Array.getShort(h5Item.getData(), 0);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String readScalarString() throws IOException {
        try {
            // Scalar Strings are String 1 dimension arrays
            Object data = h5Item.getData();
            return ((String[]) data)[0];
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean removeAttribute(final IAttribute a) {
        return HdfObjectUtils.removeAttribute(h5Item, a);
    }

    @Override
    public void setCachedData(final IArray cacheData, final boolean isMetadata) throws InvalidArrayTypeException {
        array = cacheData;
        dirty = true;
    }

    @Override
    public void setCaching(final boolean caching) {
        throw new NotImplementedException();
    }

    @Override
    public void setDataType(final Class<?> dataType) {
        throw new NotImplementedException();
    }

    @Override
    public void setDimensions(final String dimString) {

    }

    @Override
    public void setDimension(final IDimension dim, final int ind) throws DimensionNotSupportedException {

    }

    @Override
    public void setElementSize(final int elementSize) {
        throw new NotImplementedException();
    }

    @Override
    public void setSizeToCache(final int sizeToCache) {
        throw new NotImplementedException();
    }

    @Override
    public void setUnitsString(final String units) {
        throw new NotImplementedException();
    }

    @Override
    public String toString() {
        StringBuffer buffer = new StringBuffer();
        buffer.append("DataItem = " + getName());
        return buffer.toString();
    }

    @Override
    public String toStringDebug() {
        StringBuffer strDebug = new StringBuffer();
        strDebug.append(getName());
        if (strDebug.length() > 0) {
            strDebug.append("\n");
        }
        try {
            strDebug.append("shape: ").append(getData().shapeToString()).append("\n");
            List<IDimension> dimensions = getDimensionList();
            for (IDimension dim : dimensions) {
                strDebug.append(dim.getCoordinateVariable().toString());
            }

            List<IAttribute> list = getAttributeList();
            if (list.size() > 0) {
                strDebug.append("\nAttributes:\n");
            }
            for (IAttribute a : list) {
                strDebug.append("- ").append(a.toString()).append("\n");
            }
        } catch (IOException e) {
        }
        return strDebug.toString();
    }

    public void save(final FileFormat fileToWrite, final Group parentInFile) throws Exception {
        boolean saveInDifferentFile = false;
        if (parentInFile == null) {
            throw new RuntimeException("Can not save a dataitem with no parent");
        }
        // Save if it's dirty
        boolean doSave = dirty;

        if (this.h5File != null) {
            saveInDifferentFile = !(fileToWrite.getAbsolutePath().equals(h5File.getAbsolutePath()));
        }

        // But save also if when we are going to write in a new file
        if (!dirty) {
            // If this dataset exists in a file ( = !dirty)
            // then we only save it if the destination file is a different file
            doSave = saveInDifferentFile;
        }
        if (doSave) {

            if (isLink) {
                HObject link = fileToWrite
                        .createLink(parentInFile, getShortName(), h5Item, HDF5Constants.H5L_TYPE_SOFT);
                List<IAttribute> attribute = getAttributeList();
                for (IAttribute iAttribute : attribute) {
                    HdfAttribute attr = (HdfAttribute) iAttribute;
                    attr.save(link, true);
                }
                return;
            }

            try {
                // Default Value
                long[] shape = { 0, 0 };

                if (array == null) {
                    array = getData();
                }

                // shape can be determined by IArray
                if (array != null) {
                    shape = HdfObjectUtils.convertIntToLong(array.getShape());
                } else if (h5Item != null) {
                    // shape can be determined by h5Item
                    shape = h5Item.getDims();
                }

                // Datatype can be determined by IArray
                Datatype datatype = null;
                if (array != null) {
                    if (array.getStorage() instanceof String[]) {
                        String[] value = (String[]) array.getStorage();
                        datatype = new H5Datatype(Datatype.CLASS_STRING, value[0].length() + 1, -1, -1);
                    } else {
                        int type_id = HdfObjectUtils.getNativeHdfDataTypeForClass(array.getElementType());
                        datatype = new H5Datatype(type_id);
                    }
                } else if (h5Item != null) {
                    // Datatype can be determined the H5 item itself
                    datatype = h5Item.getDatatype();
                } else {
                    int type_id = HdfObjectUtils.getNativeHdfDataTypeForClass(Integer.class);
                    datatype = new H5Datatype(type_id);
                }

                H5ScalarDS ds = null;

                if (!saveInDifferentFile) {
                    ds = (H5ScalarDS) fileToWrite.get(getName());

                    // ds = (H5ScalarDS) H5File.findObject(fileToWrite, getName());
                    if (ds != null && h5Item != null) {
                        ds.init();
                        if (!datatype.equals(ds.getDatatype())) {
                            try {
                                // Ouch ! Datatype has changed.
                                // We have to create a new DS.
                                fileToWrite.delete(h5Item);
                            } catch (HDF5LibraryException hdfexception) {
                                // Sometime it's impossible to delete
                                // Don't ask me why but you should contact the HDF foundation.
                            }
                        }
                    }
                }

                ds = (H5ScalarDS) fileToWrite.createScalarDS(getName(), parentInFile, datatype, shape, null, null, 0,
                        null);

                if (ds != null) {
                    if (array != null) {
                        ds.write(array.getStorage());
                    }
                }

                List<IAttribute> attribute = getAttributeList();
                for (IAttribute iAttribute : attribute) {
                    HdfAttribute attr = (HdfAttribute) iAttribute;
                    attr.save(ds, saveInDifferentFile);
                }

                if (!saveInDifferentFile) {
                    this.h5File = (H5File) fileToWrite;
                    this.h5Item = ds;
                }
                this.dirty = false;
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            List<IAttribute> attribute = getAttributeList();
            for (IAttribute iAttribute : attribute) {
                HdfAttribute attr = (HdfAttribute) iAttribute;
                attr.save(h5Item, saveInDifferentFile);
            }
        }
    }

    @Override
    public String writeCDL(final String indent, final boolean useFullName, final boolean strict) {
        throw new NotImplementedException();
    }
}
