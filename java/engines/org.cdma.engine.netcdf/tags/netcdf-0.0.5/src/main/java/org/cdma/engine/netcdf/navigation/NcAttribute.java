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

import org.cdma.Factory;
import org.cdma.engine.netcdf.array.NcArray;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;

/**
 * Netcdf implementation of GDM Array.
 * @author nxi
 * 
 */
public class NcAttribute extends ucar.nc2.Attribute implements IAttribute {

	/**
	 * The default name label.
	 */
	private static final String DEFAULT_NAME = "default";

    /**
     * Name of the instantiating factory 
     */
	private String factoryName;
	
	/**
	 * Wrapper constructor.
	 * 
	 * @param from
	 *            Netcdf Attribute object
	 */
	public NcAttribute(final ucar.nc2.Attribute from, String factoryName) {
		// super(from);
		super(from.getName(), from.getValues());
		this.factoryName = factoryName;
	}

	/**
	 * Create an empty Attribute object.
	 */
	public NcAttribute(String factoryName) {
		super(DEFAULT_NAME);
		this.factoryName = factoryName;
	}

	// public NcAttribute(Parameter param) {
	// super(param);
	// }
	//
	/**
	 * Constructor that initialise the name and value.
	 * 
	 * @param name
	 *            String value
	 * @param values
	 *            IArray object
	 */
	public NcAttribute(final String name, final IArray values) {
		super(name, ((NcArray) values).getArray());
		this.factoryName = values.getFactoryName();
	}

	/**
	 * Construct an Attribute and change the name.
	 * 
	 * @param name
	 *            String object
	 * @param from
	 *            Netcdf Attribute object
	 */
	public NcAttribute(final String name, final ucar.nc2.Attribute from, String factoryName) {
		super(name, from);
		this.factoryName = factoryName;
	}

	/**
	 * Construct an Attribute with a boolean value.
	 * 
	 * @param name
	 *            String value
	 * @param validate
	 *            true or false
	 */
	public NcAttribute(final String name, final boolean validate, String factoryName) {
		super(name, String.valueOf(validate));
		this.factoryName = factoryName;
	}

	/**
	 * Construct an Attribute with a numeric value.
	 * 
	 * @param name
	 *            String value
	 * @param val
	 *            in numeric type
	 */
	public NcAttribute(final String name, final Number val, String factoryName) {
		super(name, val);
		this.factoryName = factoryName;
	}

	/**
	 * Construct an Attribute with a String value.
	 * 
	 * @param name
	 *            String value
	 * @param val
	 *            String value
	 */
	public NcAttribute(final String name, final String val, String factoryName) {
		super(name, val);
		this.factoryName = factoryName;
	}

	/**
	 * Construct an Attribute with empty value.
	 * 
	 * @param name
	 *            String value
	 */
	public NcAttribute(final String name, String factoryName) {
		super(name);
		this.factoryName = factoryName;
	}

	@Override
	public Class<?> getType() {
		return getDataType().getClassType();
	}

	@Override
	public IArray getValue() {
		return new NcArray(getValues(), this.factoryName);
	}

	@Override
	public void setValue(final IArray value) {
		setValues(((NcArray) value).getArray());
	}

	@Override
	public void setStringValue(final String val) {
		NcArray array = (NcArray) Factory.getFactory(this.factoryName).createArray(val.toCharArray());
		setValue(array);
	}
	
	@Override
	public String getFactoryName() {
		return this.factoryName;
	}

}
