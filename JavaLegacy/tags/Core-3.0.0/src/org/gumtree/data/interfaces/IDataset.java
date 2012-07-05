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

import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.exception.GDMWriterException;

/**
 * Interface of Dataset. A Dataset is a physical storage of GDM objects. 
 * A dataset holds a reference of a root group, which is the root of a 
 * tree of Groups.
 * 
 * @author nxi
 * 
 */
public interface IDataset extends IModelObject {

	/**
	 * Close the dataset.
	 * 
	 * @throws IOException
	 *             Created on 16/06/2008
	 */
	void close() throws IOException;

	/**
	 * Return the root group of the dataset.
	 * 
	 * @return GDM Group type Created on 16/06/2008
	 */
	IGroup getRootGroup();
	
	/**
	 * Return the the logical root of the dataset.
	 * 
	 * @return GDM Group type Created on 16/06/2008
	 */
	ILogicalGroup getLogicalRoot();

	/**
	 * Return the location of the dataset. If it's a file, return the path.
	 * Otherwise return null.
	 * 
	 * @return String type Created on 16/06/2008
	 */
	String getLocation();

	/**
	 * Return the title of the dataset.
	 * 
	 * @return String type Created on 16/06/2008
	 */
	String getTitle();

	/**
	 * Set the location field of the dataset.
	 * 
	 * @param location
	 *            in String type Created on 15/07/2008
	 */
	void setLocation(String location);

	/**
	 * Set the title for the Dataset.
	 * 
	 * @param title
	 *            a String object Created on 15/07/2008
	 */
	void setTitle(String title);

	/**
	 * Synchronize the dataset with the file reference.
	 * 
	 * @return true or false
	 * @throws IOException
	 *             Created on 15/07/2008
	 */
	boolean sync() throws IOException;

	/**
	 * Open the dataset from a file reference.
	 * 
	 * @throws IOException
	 *             Created on 15/07/2008
	 */
	void open() throws IOException;

	/**
	 * Save the contents / changes of the dataset to the file.
	 * 
	 * @throws GDMWriterException
	 *             failed to write Created on 15/07/2008
	 */
	void save() throws GDMWriterException;

	/**
	 * Save the contents of the dataset to a new file.
	 * 
	 * @throws GDMWriterException
	 *             failed to write Created on 15/07/2008
	 */
	void saveTo(String location) throws GDMWriterException;

	/**
	 * Save the specific contents / changes of the dataset to the file.
	 * 
	 * @throws GDMWriterException
	 *             failed to write Created on 15/07/2008
	 */
	void save(IContainer container) throws GDMWriterException;
	
	/**
	 * Save the attribute to the specific path of the file.
	 * 
	 * @throws GDMWriterException
	 *             failed to write Created on 15/07/2008
	 */	
	void save(String parentPath, IAttribute attribute) throws GDMWriterException;
	/**
	 * Write the file with NcML format.
	 * 
	 * @param os
	 *            java i/o OutputStream
	 * @param uri
	 *            a path to the file
	 * @throws java.io.IOException
	 *             Created on 15/07/2008
	 */
	void writeNcML(java.io.OutputStream os, java.lang.String uri)
			throws java.io.IOException;

	/**
	 * Check if the data set is open.
	 * 
	 * @return true or false
	 */
	boolean isOpen();
}
