//******************************************************************************
// Copyright (c) 2013 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// See AUTHORS file
//******************************************************************************
package org.cdma.utilities.navigation.internal;

import java.util.ArrayList;
import java.util.List;

import org.cdma.IFactory;
import org.cdma.exception.BackupException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;

public abstract class NodeParentAttribute extends NodeParent {

	private List<IAttribute> mAttributes;  // Attributes of this
	private boolean mInitialized;
	
	public NodeParentAttribute(String factory, IDataset dataset, IGroup parent, String name) throws BackupException {
		super( factory, dataset, parent, name );
		mAttributes = new ArrayList<IAttribute>();
	}
	
	public NodeParentAttribute( NodeParentAttribute object ) throws BackupException {
		super( object );
		mAttributes = new ArrayList<IAttribute>( object.mAttributes );
	}
	
	public void addOneAttribute(IAttribute attribute) {
		initialize();
		mAttributes.add(attribute);
	}

	public void addStringAttribute(String name, String value) {
		initialize();
		IFactory factory = getFactory();
		IAttribute attr = factory.createAttribute(name, value);
		if( attr != null ) {
			mAttributes.add(attr);
		}
	}

	public IAttribute getAttribute(String name) {
		initialize();
		IAttribute result = null;
		if( name != null ) {
			for( IAttribute attribute : mAttributes ) {
				if( name.equals(attribute.getName()) ) {
					result = attribute;
					break;
				}
			}
		}
		return result;
	}

	public final List<IAttribute> getAttributeList() {
		initialize();
		return mAttributes;
	}

	public boolean hasAttribute(String name, String value) {
		initialize();
		boolean result = false;
		if( name != null ) {
			String attrVal;
			for( IAttribute attribute : mAttributes ) {
				attrVal = attribute.getStringValue();
				if( attribute != null && name.equals( attribute.getName() ) ) {
					if( ( value == null && attrVal == null ) || value.equals( attrVal ) ) {
						result = true;
						break;
					}
				}
			}
		}
		return result;
	}

	public boolean removeAttribute(IAttribute attribute) {
		initialize();
		return mAttributes.remove(attribute);
	}
	
	private void initialize() {
		synchronized( this ) {
			if( ! mInitialized ) {
				mInitialized = true;
				initAttributes();
			}
		}
	}
	
	/**
	 * Initialize internal values: children items and groups, attributes, dimensions
	 */
	abstract protected void initAttributes();

}
