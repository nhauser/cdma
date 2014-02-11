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
package org.cdma.plugin.xml.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
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
import org.cdma.plugin.xml.array.XmlArray;
import org.cdma.utils.Utilities.ModelType;

public class XmlDataItem extends XmlContainer implements IDataItem, Cloneable {

    private IArray mArrayValue;
    private List<IDimension> mDimensions;

    public XmlDataItem(String factory, String name, int index, IDataset dataset, IGroup parent) {
        super(factory, name, index, dataset, parent);
        mDimensions = new ArrayList<IDimension>();
        try {
            mArrayValue = new XmlArray(factory, "");
        } catch (InvalidArrayTypeException e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to initialize array!", e);
        }
    }

    public XmlDataItem(XmlDataItem copy) {
        super(copy.getFactoryName(), copy.getName(), copy.getIndex(), copy.getDataset(), copy.getParentGroup());
        setShortName(copy.getShortName());
        try {
            mArrayValue = copy.getData();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (IAttribute attribute : copy.getAttributeList()) {
            addOneAttribute(attribute);
        }
    }

    @Override
    public IDataItem clone() {
        return new XmlDataItem(this);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DataItem;
    }

    @Override
    public IAttribute findAttributeIgnoreCase(String name) {
        IAttribute result = null;
        List<IAttribute> listAttribute = getAttributeList();
        if (listAttribute != null && name != null) {
            for (IAttribute attr : listAttribute) {
                if (name.equalsIgnoreCase(attr.getName())) {
                    result = attr;
                    break;
                }
            }
        }

        return result;
    }

    @Override
    public int findDimensionIndex(String name) {
        int result = -1;

        IGroup parent = getParentGroup();
        if (parent != null && name != null) {
            int index = 0;
            for (IDimension dim : parent.getDimensionList()) {
                if (name.equals(dim.getName())) {
                    result = index;
                    break;
                }
                index++;
            }
        }

        return result;
    }

    @Override
    public IDataItem getASlice(int dimension, int value) throws InvalidRangeException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IArray getData() throws IOException {
        return mArrayValue;
    }

    @Override
    public IArray getData(int[] origin, int[] shape) throws IOException, InvalidRangeException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public String getDescription() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<IDimension> getDimensions(int index) {
        // TODO implement a real behaviour
        return mDimensions;
    }

    @Override
    public List<IDimension> getDimensionList() {
        return mDimensions;
    }

    @Override
    public String getDimensionsString() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public int getElementSize() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public String getNameAndDimensions() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void getNameAndDimensions(StringBuffer buf, boolean longName, boolean length) {
        // TODO Auto-generated method stub

    }

    @Override
    public List<IRange> getRangeList() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public int getRank() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public IDataItem getSection(List<IRange> section) throws InvalidRangeException {
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
        // TODO Auto-generated method stub
        return null;
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
        // TODO Auto-generated method stub
        return null;
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
        mArrayValue = cacheData;
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
        // TODO Auto-generated method stub

    }

    @Override
    public void setDimension(IDimension dim, int ind) throws DimensionNotSupportedException {
        // TODO Auto-generated method stub

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

    // /////////////////////
    // Debug methods
    // /////////////////////

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

    @Override
    public String toString() {
        return getName();
    }
}
