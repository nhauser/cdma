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
package org.gumtree.data.interfaces;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.gumtree.data.exception.NoResultException;
import org.gumtree.data.exception.SignalNotAvailableException;

/**
 * A Group is a logical collection of DataItems. The Groups in a Dataset form a
 * hierarchical tree, like directories on a disk. A Group has a name and
 * optionally a set of Attributes.
 * 
 * @author nxi
 * 
 */
public interface IGroup extends IContainer {

	/**
	 * Add a data item to the group.
	 * 
	 * @param v
	 *            IDataItem object
	 */
	void addDataItem(IDataItem v);

	/**
     * 
     */
	Map<String, String> harvestMetadata(final String mdStandard)
			throws IOException;

	/**
	 * Get its parent Group, or null if its the root group.
	 * 
	 * @return GDM group object
	 */
	@Override
	IGroup getParentGroup();

	/**
	 * Get the root group of the tree that holds the current Group.
	 * 
	 * @return GDM Group Created on 18/06/2008
	 */
	@Override
	IGroup getRootGroup();

	/**
	 * Add a shared Dimension.
	 * 
	 * @param dimension
	 *            GDM IDimension object
	 */
	void addOneDimension(IDimension dimension);

	/**
	 * Add a nested Group.
	 * 
	 * @param group
	 *            GDM IGroup object
	 */
	void addSubgroup(IGroup group);

	/**
	 * Find the DataItem with the specified (short) name in this group.
	 * 
	 * @param shortName
	 *            short name of DataItem within this group.
	 * @return the Variable, or null if not found
	 */
	IDataItem getDataItem(String shortName);

	/**
	 * Find the DataItem corresponding to the given key in the dictionary.
	 * 
	 * @param key
	 *            entry name of the dictionary
	 * 
	 * @return the first encountered DataItem that match the key
	 */
	IDataItem findDataItem(IKey key);

	/**
	 * Find the DataItem that has the specific attribute, with the name and
	 * value given.
	 * 
	 * @param name
	 *            in String type
	 * @param value
	 *            in String type
	 * @return DataItem object Created on 12/03/2008
	 */
	IDataItem getDataItemWithAttribute(String name, String value);

	/**
	 * Find the DataItem corresponding to the given key in the dictionary. A
	 * filter will be applied to determine the DataItem that is the most
	 * relevant.
	 * 
	 * @param key
	 *            key to look for in the dictionary
	 * @param name
	 *            name of the attribute the key should have
	 * @param value
	 *            the attribute value
	 * @return DataItem object Created on 12/03/2008
	 */
	IDataItem findDataItemWithAttribute(IKey key, String name, String attribute)
			throws Exception;

	/**
	 * Find the Group corresponding to the given key in the dictionary. The
	 * group must have given attribute name and value. A filter will be applied
	 * to determine the Group that is the most relevant.
	 * 
	 * @param key
	 *            key to look for in the dictionary
	 * @param name
	 *            name of the attribute the group must have
	 * @param value
	 *            the attribute value
	 */
	IGroup findGroupWithAttribute(IKey key, String name, String value);

	/**
	 * Retrieve a Dimension using its (short) name. If it does not exist in this
	 * group, recursively look in parent groups.
	 * 
	 * @param name
	 *            Dimension name.
	 * @return the Dimension, or null if not found
	 */
	IDimension getDimension(String name);

	/**
	 * Retrieve the IObject that has the given short name. The object can be
	 * either a group or a data item.
	 * 
	 * @param shortName
	 *            as String object
	 * @return GDM group or data item
	 */
	IContainer getContainer(String shortName);

	/**
	 * Retrieve the Group with the specified (short) name as a sub-group of the
	 * current group.
	 * 
	 * @param shortName
	 *            short name of the nested group you are looking for.
	 * @return the Group, or null if not found
	 */
	IGroup getGroup(String shortName);

	/**
	 * Find the sub-Group that has the specific attribute, with the name and
	 * value given.
	 * 
	 * @param attributeName
	 *            String object
	 * @param value
	 *            in String type
	 * @return Group object Created on 12/03/2008
	 */
	IGroup getGroupWithAttribute(String attributeName, String value);

	/**
	 * Get the DataItem by searching the path in the dictionary with the given
	 * name. The target DataItem is not necessary to be under the current data
	 * item. If there are more than one paths associated with the same key word,
	 * use the order of their appearance in the dictionary to find the first not
	 * null object. If there is an entry wildcard, it will return the data item
	 * in the current entry.
	 * 
	 * @param shortName
	 *            in String type
	 * @return GDM DataItem Created on 18/06/2008
	 */
	IDataItem findDataItem(String shortName);

	/**
	 * Get the Variables contained directly in this group.
	 * 
	 * @return List of type Variable; may be empty, not null.
	 */
	List<IDataItem> getDataItemList();

	/**
	 * Get the Dataset that hold the current Group.
	 * 
	 * @return GDM Dataset Created on 18/06/2008
	 */
	IDataset getDataset();

	/**
	 * Get the Dimensions contained directly in this group.
	 * 
	 * @return List of type Dimension; may be empty, not null.
	 */
	List<IDimension> getDimensionList();

	/**
	 * Get the Group by searching the path in the dictionary with the given
	 * name. The target Group is not necessary to be under the current Group. If
	 * there are more than one paths associated with the key word, find the
	 * first not null group in these paths.
	 * 
	 * @param shortName
	 *            in String type
	 * @return GDM Group Created on 18/06/2008
	 */
	IGroup findGroup(String shortName);

	/**
	 * Find the Group corresponding to the given key in the dictionary.
	 * 
	 * @param key
	 *            entry name of the dictionary
	 */
	IGroup findGroup(IKey key);

	/**
	 * Get the Groups contained directly in this Group.
	 * 
	 * @return List of type Group; may be empty, not null.
	 */
	List<IGroup> getGroupList();

	/**
	 * Find the Object by searching the path in the dictionary with the given
	 * name. The target Object is not necessary to be under the current Group.
	 * The Object can be a GDM Group, GDM DataItem. If there are more than one
	 * paths associated with the same key word, use the order of their
	 * appearance in the dictionary to find the first not null object. If there
	 * is an entry wildcard, it will return the object in the current entry.
	 * 
	 * @param shortName
	 *            in String type
	 * @return IObject Created on 18/06/2008
	 */
	IContainer findContainer(String shortName);

	/**
	 * Get the Object by searching the path in the root group. The target Object
	 * is not necessary to be under the current Group. The Object can be a GDM
	 * Group, GDM DataItem, or GDM Attribute
	 * 
	 * @param path
	 *            full path of the object in String type
	 * @return GDM object Created on 13/10/2008
	 */
	IContainer findContainerByPath(String path) throws NoResultException;

	/**
	 * Get all Object by searching the path from the root group. Targeted
	 * Objects are not necessary to be directly under the current Group. Objects
	 * can be a GDM Groups, GDM DataItems, or GDM Attributes
	 * 
	 * @param path
	 *            full path of objects in String type
	 * @return GDM object Created on 29/03/2011
	 */
	List<IContainer> findAllContainerByPath(String path) throws NoResultException;

	/**
	 * Remove a DataItem from the DataItem list.
	 * 
	 * @param item
	 *            GDM DataItem
	 * @return boolean type Created on 18/06/2008
	 */
	boolean removeDataItem(IDataItem item);

	/**
	 * remove a Variable using its (short) name, in this group only.
	 * 
	 * @param varName
	 *            Variable name.
	 * @return true if Variable found and removed
	 */
	boolean removeDataItem(String varName);

	/**
	 * remove a Dimension using its name, in this group only.
	 * 
	 * @param dimName
	 *            Dimension name
	 * @return true if dimension found and removed
	 */
	boolean removeDimension(String dimName);

	/**
	 * Remove a Group from the sub Group list.
	 * 
	 * @param group
	 *            GDM Group
	 * @return boolean type Created on 18/06/2008
	 */
	boolean removeGroup(IGroup group);

	/**
	 * Remove the Group with a certain name in the sub Group list.
	 * 
	 * @param shortName
	 *            in String type
	 * @return boolean type Created on 18/06/2008
	 */
	boolean removeGroup(String shortName);

	/**
	 * Remove a Dimension from the Dimension list.
	 * 
	 * @param dimension
	 *            GDM Dimension
	 * @return GDM Dimension type Created on 18/06/2008
	 */
	boolean removeDimension(IDimension dimension);

	/**
	 * Update the data item in the location labeled by the key with a new data
	 * item. If the previous data item labeled by the key doesn't exist, it
	 * will put the data item in the location. This will also update the parent
	 * reference of the data item to the new one. If the key can not be found in
	 * the dictionary, or the parent path referred by the dictionary doesn't
	 * exist, raise an exception.
	 * 
	 * @param key
	 *            in String type
	 * @param dataItem
	 *            DataItem object Created on 17/03/2009
	 * @throws SignalNotAvailableException
	 *             no signal exception
	 */
	void updateDataItem(String key, IDataItem dataItem)
			throws SignalNotAvailableException;

	/**
	 * Set a dictionary to the root group.
	 * 
	 * @param dictionary
	 *            the dictionary to set
	 */
	void setDictionary(IDictionary dictionary);

	/**
	 * Get the dictionary from the root group.
	 * 
	 * @return IDictionary object
	 */
	IDictionary findDictionary();

	/**
	 * Check if this is the root group.
	 * 
	 * @return true or false
	 */
	boolean isRoot();

	/**
	 * Check if this is an entry group. Entries are immediate sub-group of the
	 * root group.
	 * 
	 * @return true or false
	 */
	boolean isEntry();

	/**
	 * The GDM dictionary allows multiple occurrences of a single key. This
	 * method find all the objects referenced by the given key string. If there
	 * is an entry wildcard, it will return the objects in the current entry.
	 * 
	 * @param key
	 *            Key object
	 * @return a list of GDM objects
	 */
	List<IContainer> findAllContainers(IKey key) throws NoResultException;

	/**
	 * Find all the occurrences of objects referenced by the first path for the
	 * given key in the dictionary. Those occurrences are from the available
	 * entries of the root group.
	 * 
	 * @param key
	 *            Key object
	 * @return a list of GDM objects
	 * @throws NoResultException 
	 */
	List<IContainer> findAllOccurrences(IKey key) throws NoResultException;

	/**
	 * Return a clone of this Group object. The tree structure is new. However
	 * the data items are shallow copies that share the same storages with the
	 * original ones.
	 * 
	 * @return new Group GDM group object Created on 18/09/2008
	 */
	@Override
	IGroup clone();

}
