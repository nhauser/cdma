/*******************************************************************************
 * Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 ******************************************************************************/
package org.gumtree.data.interfaces;

import java.util.List;

import org.gumtree.data.utils.Utilities.ModelType;

/**
 * Shared interface between Groups and DataItems.
 * 
 * @author nxi
 * 
 */
public interface IContainer extends IModelObject, Cloneable {
    
	/**
	 * Get the ModelType implemented by this object.
	 * 
	 * @return ModelType of this object Created on 02/04/2010
	 */
	ModelType getModelType();

	/**
	 * Add an Attribute to the Group.
	 * 
	 * @param attribute
	 *            GDM Attribute Created on 18/06/2008
	 */
	void addOneAttribute(IAttribute attribute);

	/**
	 * A convenience method of adding a String type attribute.
	 * 
	 * @param name
	 *            String type object
	 * @param value
	 *            String type object Created on 06/03/2008
	 */
	void addStringAttribute(String name, String value);

	/**
	 * Find an Attribute in this Group by its name.
	 * 
	 * @param name
	 *            the name of the attribute
	 * @return the attribute, or null if not found
	 */
	IAttribute getAttribute(String name);

	/**
	 * Get the set of attributes contained directly in this Group.
	 * 
	 * @return List of type Attribute; may be empty, not null.
	 */
	List<IAttribute> getAttributeList();

	/**
	 * Get the Dataset that hold the current Group.
	 * 
	 * @return GDM Dataset Created on 18/06/2008
	 */
	IDataset getDataset();

	/**
	 * Get the location referenced by the Dataset.
	 * 
	 * @return String type Created on 18/06/2008
	 */
	String getLocation();

	/**
	 * Get the (long) name of the IObject, which contains the path information.
	 * 
	 * @return String type object Created on 18/06/2008
	 */
	String getName();

	/**
	 * Get its parent Group, or null if its the root group.
	 * 
	 * @return GDM group object
	 */
	IContainer getParentGroup();

	/**
	 * Get the root group of the tree that holds the current Group.
	 * 
	 * @return GDM Group Created on 18/06/2008
	 */
	IContainer getRootGroup();

	/**
	 * Get the "short" name, unique within its parent Group.
	 * 
	 * @return String object
	 */
	String getShortName();

	/**
	 * Check if the Group has an Attribute with certain name and value.
	 * 
	 * @param name
	 *            in String type
	 * @param value
	 *            in String type
	 * @return boolean type Created on 18/06/2008
	 */
	boolean hasAttribute(String name, String value);

	/**
	 * Remove an Attribute from the Attribute list.
	 * 
	 * @param attribute
	 *            GDM Attribute
	 * @return boolean type Created on 18/06/2008
	 */
	boolean removeAttribute(IAttribute attribute);

	/**
	 * Set the IObject's (long) name.
	 * 
	 * @param name
	 *            String object
	 */
	void setName(String name);

	/**
	 * Set the IObject's (short) name.
	 * 
	 * @param name
	 *            in String type Created on 18/06/2008
	 */
	void setShortName(String name);

	/**
	 * Set the parent group.
	 * 
	 * @param group
	 *            IGroup object
	 */
	void setParent(IGroup group);
	

    /**
     * Clone this IContainer.
     * 
     * @return new DataItem instance
     */
    IContainer clone() throws CloneNotSupportedException;

}
