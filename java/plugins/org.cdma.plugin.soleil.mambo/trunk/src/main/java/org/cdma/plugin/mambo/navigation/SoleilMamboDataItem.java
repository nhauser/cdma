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
package org.cdma.plugin.mambo.navigation;

import java.io.IOException;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.archiving.navigation.ArchivingDataItem;
import org.cdma.exception.BackupException;
import org.cdma.exception.DimensionNotSupportedException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.plugin.mambo.SoleilMamboFactory;
import org.cdma.plugin.xml.navigation.XmlDataItem;
import org.cdma.utilities.navigation.AbstractDataItem;

public class SoleilMamboDataItem extends AbstractDataItem {
	private ArchivingDataItem item;
	
	public SoleilMamboDataItem(SoleilMamboDataset dataset, SoleilMamboGroup parent, XmlDataItem item) throws BackupException {
		super(SoleilMamboFactory.NAME, dataset, parent, item.getShortName(), null );
		
		try {
			this.setCachedData(item.getData(), false);
		} catch (InvalidArrayTypeException e) {
		} catch (IOException e) {
		}
		
		for( IAttribute attribute : item.getAttributeList() ) {
			addOneAttribute(attribute);
		}
	}
	
	public SoleilMamboDataItem(SoleilMamboDataset dataset, SoleilMamboGroup parent, ArchivingDataItem item) throws BackupException {
		super(SoleilMamboFactory.NAME, dataset, parent, item.getShortName(), null );
		this.item = item;
		for( IAttribute attribute : this.item.getAttributeList() ) {
			addOneAttribute(attribute);
		}
	}
	
	public SoleilMamboDataItem(SoleilMamboDataset dataset, SoleilMamboGroup parent, String name, IArray data) throws BackupException {
		super(SoleilMamboFactory.NAME, dataset, parent, name, data);
		if( parent != null ) {
			int i = 0;
			for( IDimension dimension : parent.getDimensionList() ) {
				try {
					setDimension(dimension, i);
					i++;
				} catch (DimensionNotSupportedException e) {
					throw new BackupException("Unable to init dimensions!", e);
				}
			}
		}
	}

	@Override
	public SoleilMamboDataItem clone() {
		SoleilMamboDataItem result = null;
		
		try {
			result = new SoleilMamboDataItem((SoleilMamboDataset) getDataset(), (SoleilMamboGroup) getParentGroup(), getShortName(), getData() );
		} catch (BackupException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to clone!", e);
		} catch (IOException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to clone!", e);
		}
		
		return result;
	}

	@Override
	public long getLastModificationDate() {
		long result = 0;
		
		IDataset dataset = getDataset();
		if( dataset != null ) {
			result = dataset.getLastModificationDate();
		}
		
		return result;
	}

	@Override
	protected void initAttributes() {
		// Nothing to do
	}
	
	@Override
	public IArray getData() throws IOException {
		IArray result = super.getData();
		
		if( result == null && item != null ) {
			result = item.getData();
			try {
				setCachedData(result, false);
			} catch (InvalidArrayTypeException e) {
			}
		}
		
		return result;
	}
	
	@Override
	public String toString() {
		StringBuffer result = new StringBuffer();
		result.append( getName() );
		result.append("\nattrib: \n" );
		for( IAttribute attr : getAttributeList() ) {
			result.append("  - " + attr.toString() + "\n" );
		}
		return result.toString();
	}
	
}
