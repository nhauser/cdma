//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.xml.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import org.cdma.dictionary.Path;
import org.cdma.exception.NoResultException;
import org.cdma.exception.SignalNotAvailableException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.utils.Utilities.ModelType;

public class XmlGroup extends XmlContainer implements IGroup, Cloneable {

	public static final String DEFAULT_SEPARATOR = "/";

	protected final SortedMap<String, IGroup> mGroup;
	protected final SortedMap<String, IDataItem> mDataItem;
	protected final List<IDimension> mDimensions;

	private String mSeparator;

	public XmlGroup(String factory, String name, int index, IDataset dataset,
			IGroup parent) {
		super(factory, name, index, dataset, parent);
		XmlIndexComparator comparator = new XmlIndexComparator();
		mGroup = new TreeMap<String, IGroup>(comparator);
		mDataItem = new TreeMap<String, IDataItem>(comparator);
		mSeparator = DEFAULT_SEPARATOR;
		mDimensions = new ArrayList<IDimension>();
	}

	public XmlGroup(XmlGroup copy) {
		this(copy.getFactoryName(), copy.getName(), copy.getIndex(), copy
				.getDataset(), copy.getParentGroup());
		setShortName(copy.getShortName());
		for (IAttribute attribute : getAttributeList()) {
			addOneAttribute(attribute);
		}
		for (IDataItem item : copy.getDataItemList()) {
			addDataItem(item);
		}
		for (IGroup group : copy.getGroupList()) {
			addSubgroup(group);
		}
	}

	@Override
	public XmlGroup clone() {
		return new XmlGroup(this);
	}

	@Override
	public ModelType getModelType() {
		return ModelType.Group;
	}

	public void addContainer(XmlContainer container) {
		if (container instanceof IGroup) {
			addSubgroup((IGroup) container);
		} else if (container instanceof IDataItem) {
			addDataItem((IDataItem) container);
		}
	}

	@Override
	public void addDataItem(IDataItem item) {
		if (item != null) {
			mDataItem.put(item.getShortName(), item);
		}
	}

	@Override
	public void addSubgroup(IGroup group) {
		if (group != null) {
			mGroup.put(group.getShortName(), group);
		}
	}

	@Override
	public IDataItem getDataItem(String shortName) {
		IDataItem result = null;
		if (shortName != null) {
			result = mDataItem.get(shortName);
			if (result == null) {
				for (IGroup group : mGroup.values()) {
					result = group.getDataItem(shortName);
					if (result != null) {
						break;
					}
				}
			}
		}
		return result;
	}

	@Override
	public IDataItem getDataItemWithAttribute(String name, String value) {
		IDataItem result = null;
		if (name != null && value != null) {
			for (IDataItem item : mDataItem.values()) {
				IAttribute attr = item.getAttribute(name);
				if (attr.getStringValue().equalsIgnoreCase(value)) {
					result = item;
					break;
				}
			}
		}
		return result;
	}

	@Override
	public IContainer getContainer(String shortName) {
		IContainer result = null;
		if (shortName != null) {
			result = mGroup.get(shortName);
			if (result == null) {
				result = mDataItem.get(shortName);
			}
		}
		return result;
	}

	@Override
	public IGroup getGroup(String shortName) {
		IGroup result = null;
		if (shortName != null) {
			result = mGroup.get(shortName);
		}
		return result;
	}

	@Override
	public IGroup getGroupWithAttribute(String attributeName, String value) {
		IGroup result = null;
		if (attributeName != null && value != null) {
			for (IGroup group : mGroup.values()) {
				IAttribute attr = group.getAttribute(attributeName);
				if (attr.getStringValue().equalsIgnoreCase(value)) {
					result = group;
					break;
				}
			}
		}
		return result;
	}

	@Override
	public List<IDataItem> getDataItemList() {
		return new ArrayList<IDataItem>(mDataItem.values());
	}

	@Override
	public List<IGroup> getGroupList() {
		return new ArrayList<IGroup>(mGroup.values());
	}

	@Override
	public boolean removeDataItem(IDataItem item) {
		boolean result = false;
		if (item != null) {
			String key = null;
			Set<Entry<String, IDataItem>> entries = mDataItem.entrySet();
			for (Entry<String, IDataItem> entry : entries) {
				if (item.equals(entry.getValue())) {
					key = entry.getKey();
				}
			}
			IDataItem removedItem = mDataItem.remove(key);
			result = (removedItem != null);
		}
		return result;
	}

	@Override
	public boolean removeDataItem(String varName) {
		boolean result = false;
		if (varName != null) {
			IDataItem removedItem = mDataItem.remove(varName);
			result = (removedItem != null);
		}
		return result;
	}

	@Override
	public boolean removeGroup(IGroup group) {
		boolean result = false;
		if (group != null) {
			String key = null;
			Set<Entry<String, IGroup>> entries = mGroup.entrySet();
			for (Entry<String, IGroup> entry : entries) {
				if (group.equals(entry.getValue())) {
					key = entry.getKey();
				}
			}
			IGroup removedItem = mGroup.remove(key);
			result = (removedItem != null);
		}
		return result;
	}

	@Override
	public boolean removeGroup(String name) {
		boolean result = false;
		if (name != null) {
			IGroup removedItem = mGroup.remove(name);
			result = (removedItem != null);
		}
		return result;
	}

	@Override
	public boolean isRoot() {
		return (getParentGroup() == null);
	}

	@Override
	public boolean isEntry() {
		IGroup parentGroup = getParentGroup();
		if (parentGroup != null) {
			parentGroup = parentGroup.getParentGroup();
			if (parentGroup != null && parentGroup instanceof IGroup) {
				return parentGroup.isRoot();
			}
		}
		return false;
	}

	// /////////////////////
	// Override methods
	// /////////////////////

	@Override
	public IGroup getRootGroup() {
		return super.getRootGroup();
	}

	@Override
	public IGroup getParentGroup() {
		IGroup result = null;
		IContainer parentGroup = super.getParentGroup();
		if (parentGroup instanceof IGroup) {
			result = (IGroup) parentGroup;
		}
		return result;
	}

	public void setSeparator(String separator) {
		this.mSeparator = separator;
	}

	public String getSeparator() {
		return mSeparator;
	}

	// /////////////////////
	// Find group methods
	// /////////////////////

	@Override
	public IContainer findContainerByPath(String path) {
		IContainer result = null;
		List<IContainer> containers = findAllContainerByPath(path);
		String[] pathElement = path.split(getSeparator());
		if (pathElement != null && pathElement.length > 0) {
			String referent = pathElement[pathElement.length - 1];
			for (IContainer container : containers) {
				if (referent.equalsIgnoreCase(container.getShortName())) {
					result = container;
					break;
				}
			}
		}
		return result;
	}

	private List<IContainer> findContainerIntoGroup(String[] pathElements) {
		List<IContainer> result = new ArrayList<IContainer>();
		if (pathElements != null && pathElements.length > 0) {
			if (pathElements.length > 1) {
				IGroup group = getGroup(pathElements[0]);
				if (group != null && group instanceof XmlGroup) {
					result.add(group);

					String[] nextPath = Arrays.copyOfRange(pathElements, 1,
							pathElements.length);
					result.addAll(((XmlGroup) group)
							.findContainerIntoGroup(nextPath));
				}
			} else {
				IDataItem item = getDataItem(pathElements[0]);
				if (item != null) {
					result.add(item);
				} else {
					result.addAll(getDataItemList());
				}
			}
		}
		return result;
	}

	@Override
	public List<IContainer> findAllContainerByPath(String path) {
		List<IContainer> result = new ArrayList<IContainer>();
		IGroup rootGroup = (isRoot()) ? this : getRootGroup();
		String[] pathElement = path.split(getSeparator());
		if (rootGroup != null) {
			result = ((XmlGroup) rootGroup).findContainerIntoGroup(pathElement);
		}
		return result;
	}

	@Override
	public IContainer findObjectByPath(Path path) {
		String pathString = path.getValue();
		return findContainerByPath(pathString);
	}

	// /////////////////////
	// Unimplemented methods
	// /////////////////////

	@Override
	public Map<String, String> harvestMetadata(String mdStandard)
			throws IOException {
		return null;
	}

	@Override
	public void addOneDimension(IDimension dimension) {
		mDimensions.add(dimension);
	}

	@Override
	public IDataItem findDataItem(IKey key) {
		return null;
	}

	@Override
	public IDataItem findDataItemWithAttribute(IKey key, String name,
			String attribute) throws NoResultException {
		return null;
	}

	@Override
	public IGroup findGroupWithAttribute(IKey key, String name, String value) {
		return null;
	}

	@Override
	public IDimension getDimension(String name) {
		IDimension result = null;
		
		for( IDimension dim : mDimensions ) {
			if( dim.getName().equals( name ) ) {
				result = dim;
				break;
			}
		}
		return result;
	}

	@Override
	public IDataItem findDataItem(String shortName) {
		return null;
	}

	@Override
	public List<IDimension> getDimensionList() {
		return mDimensions;
	}

	@Override
	public IGroup findGroup(String shortName) {
		return null;
	}

	@Override
	public IGroup findGroup(IKey key) {
		return null;
	}

	@Override
	public IContainer findContainer(String shortName) {
		return null;
	}

	@Override
	public boolean removeDimension(String name) {
		return false;
	}

	@Override
	public boolean removeDimension(IDimension dimension) {
		return false;
	}

	@Override
	public void updateDataItem(String key, IDataItem dataItem)
			throws SignalNotAvailableException {
	}

	@Override
	public void setDictionary(IDictionary dictionary) {
	}

	@Override
	public IDictionary findDictionary() {
		return null;
	}

	@Override
	public List<IContainer> findAllContainers(IKey key)
			throws NoResultException {
		return null;
	}

	@Override
	public List<IContainer> findAllOccurrences(IKey key)
			throws NoResultException {
		return null;
	};

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
		System.out.print(">\n");

		// Subgroup description
		for (IGroup group : mGroup.values()) {
			if (group instanceof XmlGroup) {
				((XmlGroup) group).printHierarchy(level + 1);
			}
		}

		// Items description
		for (IDataItem item : mDataItem.values()) {
			if (item instanceof XmlDataItem) {
				((XmlDataItem) item).printHierarchy(level + 1);
			}
		}

		// Tag Ending
		System.out.print(tabFormatting);
		System.out.print("</");
		System.out.print(getShortName());
		System.out.print(">\n");
	}

	public class XmlIndexComparator implements Comparator<String> {

		@Override
		public int compare(String s1, String s2) {
			int firstIndex = retrieveContainerIndex(s1);
			int secondIndex = retrieveContainerIndex(s2);
			if (firstIndex == -1 && secondIndex == -1) {
				return s1.compareTo(s2);
			}
			if (firstIndex > secondIndex) {
				return 1;
			} else if (firstIndex < secondIndex) {
				return -1;
			}
			return 0;
		}
	}

}
