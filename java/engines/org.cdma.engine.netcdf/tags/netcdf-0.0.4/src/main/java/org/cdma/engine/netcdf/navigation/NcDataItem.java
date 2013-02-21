/*******************************************************************************
 * Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 ******************************************************************************/
package org.cdma.engine.netcdf.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.netcdf.array.NcArray;
import org.cdma.engine.netcdf.array.NcRange;
import org.cdma.exception.DimensionNotSupportedException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IRange;
import org.cdma.utils.Utilities.ModelType;

import ucar.ma2.DataType;
import ucar.ma2.Range;
import ucar.nc2.Attribute;
import ucar.nc2.Dimension;
import ucar.nc2.Variable;
import ucar.nc2.dataset.NetcdfDataset;
import ucar.nc2.dataset.VariableDS;

/**
 * GDM DataItem implementation as an extension of Netcdf Variable class.
 * 
 * @author nxi
 * 
 */

public class NcDataItem extends VariableDS implements IDataItem {

	/**
	 * The parent dataset.
	 */
	private NcDataset dataset;
	
    /**
     * Name of the instantiating factory 
     */
    private String factoryName;

	/**
	 * Constructor from a Netcdf VariableDS object.
	 * 
	 * @param from
	 *            Netcdf Variable
	 */
	public NcDataItem(final VariableDS from, String factoryName) {
		//[SOLEIL] 11/02/2013 changed to fit edu.ucar.netcdf version 4.2.20
		super(null, from, false);
		
		
		Cache oldCache = cache;
		cache = new Cache();
		cache.cachingSet = oldCache.cachingSet;
		cache.data = oldCache.data;
		this.factoryName = factoryName;
		// make it caching all the time
//		cache.isCaching = oldCache.isCaching;
		cache.isCaching = true;
		// setCaching(true);
		if (!(from instanceof NcDataItem)) {
			ArrayList<Attribute> newAttributes = new ArrayList<Attribute>();
			ArrayList<Dimension> newDimensions = new ArrayList<Dimension>();
			for (Iterator<?> iter = attributes.iterator(); iter.hasNext();) {
				ucar.nc2.Attribute attribute = (ucar.nc2.Attribute) iter.next();
				newAttributes.add(new NcAttribute(attribute.getName(),
						new NcArray(attribute.getValues(), factoryName)));
			}
			for (Iterator<?> iter = dimensions.iterator(); iter.hasNext();) {
				ucar.nc2.Dimension dimension = (ucar.nc2.Dimension) iter.next();
				newDimensions.add(new NcDimension(dimension.getName(),
						dimension, factoryName));
			}
			attributes = newAttributes;
			dimensions = newDimensions;
		}
	}

	/**
	 * Constructor with a parent group, name and value storage.
	 * 
	 * @param group
	 *            GDM group object
	 * @param shortName
	 *            String value
	 * @param array
	 *            GDM Array object
	 * @throws InvalidArrayTypeException
	 *             Array type is wrong
	 */
	public NcDataItem(final NcGroup group, final String shortName,
			final IArray array) throws InvalidArrayTypeException {
		super(null, group, null, shortName, DataType.getType(array
				.getElementType()), null, null, null);
		// group.insertVariable(this);
		setDataType(array.getElementType());
		setCachedData(array, true);
		createDimension(array);
		this.dataset = group.getDataset();
	}

	/**
	 * Constructor with full parameter list.
	 * 
	 * @param ncDataset
	 *            a GDM Dataset object
	 * @param group
	 *            a GDM Group object
	 * @param shortName
	 *            String value
	 * @param array
	 *            a GDM Array object
	 * @throws InvalidArrayTypeException
	 *             array type is wrong
	 */
	public NcDataItem(final NcDataset ncDataset, final NcGroup group,
			final String shortName, final IArray array)
			throws InvalidArrayTypeException {
		super((NetcdfDataset) ncDataset.getNetcdfDataset(), group, null,
				shortName, DataType.getType(array.getElementType()), null,
				null, null);
		setDataType(array.getElementType());
		// group.insertVariable(this);
		setCachedData(array, true);
		createDimension(array);
		this.dataset = ncDataset;
	}

	@Override
	public ModelType getModelType() {
		return ModelType.DataItem;
	}
	
	/**
	 * Create dimension from an array storage.
	 * 
	 * @param array
	 *            GDM Array object
	 */
	private void createDimension(final IArray array) {
		int[] shape = array.getShape();
		List<Dimension> dimensionList = new ArrayList<Dimension>();
		for (int i = 0; i < shape.length; i++) {
			if (shape[i] > 0) {
				NcDimension dimension = new NcDimension(String.valueOf(shape[i]),
						shape[i], factoryName);
				dimensionList.add(dimension);
				NcDimension groupDimension = (NcDimension) group
				.findDimension(dimension.getName());
				if (groupDimension == null) {
					group.addDimension(dimension);
				}
			}
		}
		setDimensions(dimensionList);
	}

	/**
	 * Modified _read method. Adapted to always read data in cache.
	 * 
	 * @return a Netcdf array
	 * @throws IOException
	 *             I/O error
	 */
	@Override
	protected ucar.ma2.Array _read() throws IOException {
		if (cache != null && cache.data != null) {
			if (debugCaching) {
				Factory.getLogger().log(Level.INFO, "got data from cache " + getName());
			}
			return cache.data;
		} else {
			setCachedData(super._read(), false);
			return cache.data;
		}
	}

	@Override
	public NcArray getData() throws IOException {
		return new NcArray(read(), factoryName);
	}

	@Override
	public boolean isCaching() {
		return true;
	}
	
	@Override
	public IArray getData(final int[] origin, final int[] shape)
			throws IOException, InvalidRangeException {
		try {
			return new NcArray(read(origin, shape), factoryName);
		} catch (ucar.ma2.InvalidRangeException e) {
			throw new InvalidRangeException(e);
		}
	}

	/**
	 * Set the units of the data item as String.
	 * 
	 * @param units
	 *            String value
	 */
	public void setUnits(final String units) {
		NcAttribute unitsAttribute = new NcAttribute("units", units);
		this.addOneAttribute(unitsAttribute);
	}

	@Override
	public NcAttribute getAttribute(final String name) {
		Attribute attribute = super.findAttribute(name);
		if (attribute instanceof NcAttribute) {
			return (NcAttribute) attribute;
		}
		return null;
	}

	@Override
	public void setCachedData(final IArray cacheData, final boolean isMetadata)
			throws InvalidArrayTypeException {
		if (cacheData instanceof NcArray) {
			super.setCachedData(((NcArray) cacheData).getArray(), isMetadata);
		} else {
			throw new InvalidArrayTypeException("not a netcdf Array");
		}
	}

	@Override
	public void addOneAttribute(final IAttribute att) {
		if (att instanceof NcAttribute) {
			super.addAttribute((NcAttribute) att);
		}
	}

	@Override
	public void setParent(final IGroup group) {
		if (group instanceof NcGroup) {
			setParentGroup((NcGroup) group);
		}
	}

	@Override
	public boolean removeAttribute(final IAttribute a) {
		if (a instanceof NcAttribute) {
			return super.remove((NcAttribute) a);
		}
		return false;
	}

	@Override
	public void setDataType(final Class<?> dataType) {
		super.setDataType(ucar.ma2.DataType.getType(dataType));
	}

	@Override
	public Class<?> getType() {
		return getDataType().getClassType();
	}

	@Override
	public NcDataItem getSlice(final int dim, final int value)
			throws InvalidRangeException {
		try {
			return (NcDataItem) super.slice(dim, value);
		} catch (ucar.ma2.InvalidRangeException ex) {
			throw new InvalidRangeException(ex.getMessage());
		}
	}

	@Override
	public NcDataItem getSection(final List<IRange> section)
			throws InvalidRangeException {
		List<Range> ncRangeList = new ArrayList<Range>();
		for (IRange range : section) {
			ncRangeList.add(((NcRange) range).getNetcdfRange());
		}
		try {
			return (NcDataItem) super.section(ncRangeList);
		} catch (Exception e) {
			throw new InvalidRangeException(e.getMessage());
		}
	}

	@Override
	public NcGroup getParentGroup() {
		return (NcGroup) super.getParentGroup();
	}
	
	@Override
	public List<IDimension> getDimensions(final int i) {
        List<IDimension> list = new ArrayList<IDimension>();
        list.add((NcDimension) super.getDimension(i));
        return list;
	}

	@Override
	public NcAttribute findAttributeIgnoreCase(final String name) {
		return (NcAttribute) super.findAttributeIgnoreCase(name);
	}

	/**
	 * Find the unit attribute of the variable and retrieve the String value.
	 * 
	 * @return unit in String type.
	 */
	public String getUnits() {
		NcAttribute unitAttribute = getAttribute("units");
		if (unitAttribute == null) {
			return "";
		}
		return unitAttribute.getStringValue();
	}

	@Override
	public NcDataset getDataset() {
		return dataset;
	}

	/**
	 * Set the dataset holder of the data item.
	 * 
	 * @param ncDataset
	 *            NcDataset object
	 */
	public void setDataset(final NcDataset ncDataset) {
		this.dataset = ncDataset;
		if (ncfile == null) {
			ncfile = ncDataset.getNetcdfDataset();
		}
	}

	@Override
	public NcDataItem clone() {
		return new NcDataItem(this, factoryName);
	}

	@Override
	public String toString() {
		String result = "";
		result += "<DataItem>" + getShortName() + "\n";
		// try {
		// result += "value = " + getData().toString() + "\n";
		// } catch (IOException e) {
		// // e.printStackTrace();
		// result += "value = null\n";
		// }
		List<?> attributeList = getAttributes();
		for (Iterator<?> iterator = attributeList.iterator(); iterator
				.hasNext();) {
			NcAttribute attribute = (NcAttribute) iterator.next();
			result += attribute.toString() + "\n";
		}
		result += "</DataItem>\n";
		return result;
	}

	@Override
	public void addStringAttribute(final String name, final String value) {
		NcAttribute attribute = null;
		if (value == null) {
			attribute = new NcAttribute(name, "");
		} else {
			attribute = new NcAttribute(name, value);
		}
		addAttribute(attribute);
	}

	@Override
	public NcDataItem getASlice(final int dimension, final int value)
			throws InvalidRangeException {
		NcDataItem variable = null;
		try {
			variable = new NcDataItem((VariableDS) slice(dimension, value), factoryName);
		} catch (ucar.ma2.InvalidRangeException e) {
			// e.printStackTrace();
			throw new InvalidRangeException("dimension out of boundary");
		}
		return variable;
	}

	/**
	 * Enhance the data item.
	 * 
	 * @param group
	 *            Netcdf Group object
	 * @return Netcdf Variable object
	 * @see Variable#
	 */
	Variable enhance(final ucar.nc2.Group group) {
		return new VariableDS(getParentGroup(), this, true);
	}

	@Override
	public boolean hasAttribute(final String name, final String value) {
		IAttribute attribute = getAttribute(name);
		if (attribute == null) {
			return false;
		}
		if (attribute.getStringValue().equals(value)) {
			return true;
		}
		return false;
	}

	@Override
	public void getNameAndDimensions(final StringBuffer buf,
			final boolean useFullName, final boolean showDimLength) {
	}

	@Override
	public List<IRange> getSectionRanges() {
		return null;
	}

	// @Override
	// public void setDimensionsAnonymous(final int[] shape) {
	// try {
	// super.setDimensionsAnonymous(shape);
	// } catch (ucar.ma2.InvalidRangeException e) {
	//			
	// }
	// }

	@Override
	public List<IAttribute> getAttributeList() {
		if (getAttributes() == null) {
			return null;
		}
		List<IAttribute> attributeList = new ArrayList<IAttribute>();
		for (Attribute attribute : getAttributes()) {
			attributeList.add((NcAttribute) attribute);
		}
		return attributeList;
	}

	@Override
	public List<IDimension> getDimensionList() {
		if (getDimensions() == null) {
			return null;
		}
		List<IDimension> dimensionList = new ArrayList<IDimension>();
		for (Dimension dimension : getDimensions()) {
			dimensionList.add((NcDimension) dimension);
		}
		return dimensionList;
	}

	@Override
	public List<IRange> getRangeList() {
		if (getRanges() == null) {
			return null;
		}
		List<IRange> rangeList = new ArrayList<IRange>();
		for (Range range : getRanges()) {
			rangeList.add(new NcRange(range));
		}
		return rangeList;
	}

	@Override
	public String getLocation() {
		return getParentGroup().getLocation();
	}

	@Override
	public IGroup getRootGroup() {
		return getParentGroup().getRootGroup();
	}

	@Override
	public void setShortName(final String name) {
		super.setName(name);
	}
	
    @Override
	public void setDimension(IDimension dim, int ind) throws DimensionNotSupportedException {
        super.setDimension(ind, (NcDimension) dim);
        
    }

	@Override
    public String getFactoryName() {
        return this.factoryName;
    }

    @Override
    public long getLastModificationDate() {
        return dataset.getLastModificationDate();
    }
    
    @Override
    public int getRank() {
    	int[] shape = getShape();
    	int rank = shape.length;
    	if (shape.length == 1 && shape[0] == 1) {
            rank = 0;
        }
    	
    	return rank;
    }
    
    @Override
    public int[] getShape() {
    	/*int[] shape = super.getShape();
    	if( dataType == DataType.STRING ) {
    		shape = new int[] {1};
    	}
    	else if( dataType == DataType.CHAR ) {
    		if( super.getShape().length == 2 && super.getShape()[0] == 1 ) {
    			shape = new int[] {1};
    		}
    	}*/
    	return super.getShape();
    }
}
