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

import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IDimension;

/**
 * Netcdf implementation of GDM Dimension.
 * @author nxi
 * 
 */
public class NcDimension extends ucar.nc2.Dimension implements IDimension {

	private IArray coordinateVariable;
	
    /**
     * Name of the instantiating factory 
     */
    private String factoryName;
	
	/**
	 * Constructor from Netcdf Dimension object.
	 * 
	 * @param name
	 *            String value
	 * @param from
	 *            Netcdf object
	 */
	public NcDimension(final String name, final ucar.nc2.Dimension from, String factoryName) {
		super(name, from);
		this.factoryName = factoryName;
	}

	/**
	 * Constructor from name and length.
	 * 
	 * @param name
	 *            String value
	 * @param length
	 *            integer value
	 * @param isShared
	 *            true or false
	 */
	public NcDimension(final String name, final int length,
			final boolean isShared, String factoryName) {
		super(name, length, isShared);
		this.factoryName = factoryName;
	}

	/**
	 * Create Dimension from name and length.
	 * 
	 * @param name
	 *            String value
	 * @param length
	 *            integer value
	 */
	public NcDimension(final String name, final int length, String factoryName) {
		super(name, length, false);
		this.factoryName = factoryName;
	}

	// public void addCoordinateDataItem(final DataItem item) {
	// if (item instanceof CachedVariable)
	// addCoordinateVariable((CachedVariable) item);
	// }

	@Override
	public int compareTo(final Object o) {
		return super.compareTo(o);
	}

	@Override
	public IArray getCoordinateVariable() {
		return coordinateVariable;
	}
    
    @Override
    public void setCoordinateVariable(IArray array) {
    	coordinateVariable = array;
    }
    
	@Override
	public String getFactoryName() {
		return factoryName;
	}
	
}
