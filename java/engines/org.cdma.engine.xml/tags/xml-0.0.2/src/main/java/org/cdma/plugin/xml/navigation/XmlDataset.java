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

import java.io.File;
import java.io.IOException;
import java.util.Hashtable;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.xml.XmlUtils;
import org.cdma.plugin.xml.array.XmlArray;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class XmlDataset implements IDataset {

	private File mPath;
	private String mTitle;
	private IGroup mRootGroup;
	private boolean mIsOpen;
	private final String mFactoryName;

	public XmlDataset(String factoryName, File file) {
		mIsOpen = false;
		mPath = file;
		mTitle = file.getName();
		mFactoryName = factoryName;
		if (file.exists() && file.isFile()) {
			initFromXmlFile(file);
		}
	}

	private void initFromXmlFile(File file) {
		try {
			Node rootNode = XmlUtils.getRootNode(file);
			mRootGroup = new XmlGroup(mFactoryName, rootNode.getNodeName(), 0,
					this, null);
			loadAttributes(rootNode, (XmlGroup) mRootGroup);
			loadChildNodes(rootNode, (XmlGroup) mRootGroup);
			mIsOpen = true;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void loadChildNodes(Node node, XmlGroup parentContainer)
			throws Exception {
		if (node.hasChildNodes()) {
			NodeList childNodes = node.getChildNodes();
			int groupIndex = 0, itemIndex = 0;
			for (int index = 0; index < childNodes.getLength(); ++index) {
				Node currentChild = childNodes.item(index);
				if (XmlUtils.isAFakeNode(currentChild)) {
					continue;
				}
				String currentChildType = currentChild.getNodeName().trim();

				// Add data item when a text value is available
				String data = currentChild.getNodeValue();
				if ( data != null && ! data.trim().isEmpty() ) {
					XmlDataItem item = new XmlDataItem(mFactoryName, "data_item",
							itemIndex, this, parentContainer);
					item.setCachedData(new XmlArray( mFactoryName, data), false);
					parentContainer.addContainer(item);
					itemIndex++;
				}
				// The node is a group
				else {
					XmlGroup container = null;
					if (parentContainer instanceof IGroup) {
						container = new XmlGroup(mFactoryName, currentChildType,
								groupIndex, this, parentContainer);
						loadChildNodes(currentChild, container);
						groupIndex++;
					}
	
					if (parentContainer != null
							&& parentContainer instanceof XmlGroup) {
						parentContainer.addContainer(container);
					}
					loadAttributes(currentChild, container);
				}
			}
		}
	}

	private void loadAttributes(Node node, XmlContainer container)
			throws Exception {
		Hashtable<String, String> attributeProperties = XmlUtils
				.loadAttributes(node);
		if (!attributeProperties.isEmpty()) {
			for (String entry : attributeProperties.keySet()) {
				container.addStringAttribute(entry,
						attributeProperties.get(entry));
			}
		}
	}

	@Override
	public void close() throws IOException {

	}

	@Override
	public String getFactoryName() {
		return mFactoryName;
	}

	@Override
	public IGroup getRootGroup() {
		return mRootGroup;
	}

	@Override
	public String getLocation() {
		return mPath.getPath();
	}

	@Override
	public String getTitle() {
		return mTitle;
	}

	@Override
	public void setLocation(String location) {
		File file = new File(location);
		if (file.exists() && file.isFile()) {
			mPath = file;
			initFromXmlFile(file);
		}
	}

	@Override
	public LogicalGroup getLogicalRoot() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setTitle(String title) {
		mTitle = title;
	}

	@Override
	public boolean sync() throws IOException {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void open() throws IOException {
		// TODO Auto-generated method stub

	}

	@Override
	public void save() throws WriterException {
		// TODO Auto-generated method stub

	}

	@Override
	public void saveTo(String location) throws WriterException {
		// TODO Auto-generated method stub

	}

	@Override
	public void save(IContainer container) throws WriterException {
		// TODO Auto-generated method stub

	}

	@Override
	public void save(String parentPath, IAttribute attribute)
			throws WriterException {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean isOpen() {
		return mIsOpen;
	}

	public void printHierarchy() {
		System.out
				.print("-------------------------- HIERARCHY ---------------------------------\n");
		if (mRootGroup != null) {
			((XmlGroup) mRootGroup).printHierarchy(0);
		} else {
			System.out.print("Root is null");
		}
	}

	public static void main(String[] args) {

		File xmlFile = new File(
				"D:\\Archiving\\Mambo\\vc\\bug_mambo_example_vc.xml");
		System.out.println("File exists : " + xmlFile.exists());
		System.out.println("File is a file : " + xmlFile.isFile());

		XmlDataset dataset = new XmlDataset("XML_FACTORY", xmlFile);
		System.out.println(dataset);
		System.out.println("Path : " + dataset.getLocation());
		System.out.println("Title : " + dataset.getTitle());

		IGroup group = dataset.getRootGroup();
		System.out.println("Root group : " + group);

		System.out.println();
		dataset.printHierarchy();

	}

	@Override
	public long getLastModificationDate() {
		return mPath.lastModified();
	}
}
