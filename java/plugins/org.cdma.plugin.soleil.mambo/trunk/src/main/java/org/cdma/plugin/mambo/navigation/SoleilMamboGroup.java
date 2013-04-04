//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.mambo.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.archiving.internal.attribute.AttributePath;
import org.cdma.engine.archiving.navigation.ArchivingDataset;
import org.cdma.engine.archiving.navigation.ArchivingGroup;
import org.cdma.exception.BackupException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.mambo.SoleilMamboFactory;
import org.cdma.plugin.mambo.internal.MamboConstants;
import org.cdma.plugin.xml.navigation.XmlGroup;
import org.cdma.utilities.navigation.DefaultGroup;

public class SoleilMamboGroup extends DefaultGroup {
	private static final String PATH_SEPARATOR = "/";
	private XmlGroup       mXmlGroup;
	private ArchivingGroup mArcGroup;
	
	
	public SoleilMamboGroup(SoleilMamboDataset dataset, SoleilMamboGroup parent, String name) throws BackupException {
		super( SoleilMamboFactory.NAME, dataset, parent, name );
		mXmlGroup = null;
		mArcGroup = null;
	}
	
	public SoleilMamboGroup(SoleilMamboDataset dataset, SoleilMamboGroup parent, XmlGroup group) throws BackupException {
		super( SoleilMamboFactory.NAME, dataset, parent, group.getShortName() );
		mXmlGroup = group;
		mArcGroup = null;
	}
	
	public SoleilMamboGroup(SoleilMamboDataset dataset, SoleilMamboGroup parent, XmlGroup xmlGroup, ArchivingGroup arcGroup) throws BackupException {
		super( SoleilMamboFactory.NAME, dataset, parent, arcGroup.getShortName() );
		mXmlGroup = xmlGroup;
		mArcGroup = arcGroup;
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
	public Map<String, String> harvestMetadata(String standard) throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public IGroup clone() {
		SoleilMamboGroup result = null;
		try {
			result = new SoleilMamboGroup(
										(SoleilMamboDataset) getDataset(), 
										(SoleilMamboGroup) getParentGroup(), 
										getShortName()
									);
			result.mXmlGroup = this.mXmlGroup;
			result.mArcGroup = this.mArcGroup;
		} catch (BackupException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to create a clone!", e);
		}

		return result;
	}

	// ------------------------------------------------------------------------
	// protected methods
	// ------------------------------------------------------------------------
	@Override
	protected String getPathSeparator() {
		return PATH_SEPARATOR;
	}

	@Override
	protected void initAttributes() {
		IGroup group = null;
		if( mXmlGroup != null ) {
			group = mXmlGroup;
		}
		else if( mArcGroup != null ) {
			group = mArcGroup;
		}
		
		if( group != null ) {
			List<IAttribute> attributes = group.getAttributeList();
			for( IAttribute attribute : attributes ) {
				addOneAttribute(attribute);
			}
		}
		
	}
	
	@Override
	protected void initChildren() {
		initGroupList();
		initItemList();
	}
	
	// ------------------------------------------------------------------------
	// private methods
	// ------------------------------------------------------------------------
	private void initItemList() {
		List<IGroup> groups = new ArrayList<IGroup>();
		if( mXmlGroup != null ) {
			groups.add(mXmlGroup);
		}
		if( mArcGroup != null ) {
			groups.add(mArcGroup);
		}
		
		for( IGroup group : groups ) {
			SoleilMamboDataset dataset = (SoleilMamboDataset) getDataset();
			List<IDataItem> items = group.getDataItemList();
			try {
				for( IDataItem item : items ) {
					IArray data = item.getData();
					String name = item.getShortName();
					addDataItem( new SoleilMamboDataItem( dataset, this, name, data) );
				}
			} catch (BackupException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize item child", e);
			} catch (IOException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize item child", e);
			}
		}
	}
	
	private void initGroupList() {
		SoleilMamboDataset dataset = (SoleilMamboDataset) getDataset();
		if( mXmlGroup != null ) {
			String name;
			SoleilMamboGroup group;
			List<IGroup> xmlGroups = mXmlGroup.getGroupList();
			try {
				for( IGroup xmlGroup : xmlGroups ) {
					// If group is an 'attribute' create the archiving group
					name = xmlGroup.getShortName();
					if( name.matches(MamboConstants.ATTRIBUTE_MARKUP_REGEXP) ) {
						IAttribute attr = xmlGroup.getAttribute(MamboConstants.ATTRIBUTE_NAME);
						if( attr != null ) {
							String path = attr.getStringValue();
							group = initGroupChild(dataset, path, (XmlGroup) xmlGroup);
						}
						else {
							// Create a corresponding group
							group = new SoleilMamboGroup( dataset, this, (XmlGroup) xmlGroup );
						}
					}
					else {
						// Create a corresponding group
						group = new SoleilMamboGroup( dataset, this, (XmlGroup) xmlGroup );
					}
					
					addSubgroup( group );
					
				}
			} catch (BackupException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize group child", e);
			} catch (NoResultException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize group child", e);
			}
		}
	}
	
	private SoleilMamboGroup initGroupChild(SoleilMamboDataset dataset, String path, XmlGroup xmlGroup) throws NoResultException, BackupException {
		SoleilMamboGroup result = null;
		SoleilMamboGroup parent = this;
		SoleilMamboGroup group = null;
		
		ArchivingDataset arcDataset = dataset.getArchivingDataset();
		IGroup arcGroup = arcDataset.getRootGroup();
		
		// Split the path into nodes
		String[] nodes = path.split(AttributePath.SEPARATOR);
		
		// For each node construct a SoleilMamboGroup
		for( String node : nodes ) {
			if( node != null ) {
				group = (SoleilMamboGroup) parent.getGroup(node);
				arcGroup = arcGroup.getGroup(node);
				if( group == null ) {
					if( arcGroup == null ) {
						throw new NoResultException("Path is invalid: " + path + "\nNode wasn't found: " + node );
					}
					
					// Create the mambo group
					group = new SoleilMamboGroup(dataset, parent, arcGroup.getShortName() );
					
					// Return the first created group (and only it!) as the current children
					if( parent == this ) {
						result = group;
					}
					// Descendants are added to the first created group
					else {
						parent.addSubgroup(group);
					}
				}	
				parent = group;
			}
		}
		
		// Continue the initializing normally at last group of the archiving attribute
		group.mXmlGroup = xmlGroup;
		group.mArcGroup = (ArchivingGroup) arcGroup;

		// Add dimension to that last group
		List<IDimension> dimensions = arcGroup.getDimensionList();
		for( IDimension dimension : dimensions ) {
			group.addOneDimension(dimension);
		}
		
		return result;
	}
	
}
