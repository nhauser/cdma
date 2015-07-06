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
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    ClÃ©ment Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - API evolution
// ****************************************************************************
package org.cdma.interfaces;

import java.io.IOException;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.WriterException;
import org.cdma.internal.IModelObject;

/**
 * @brief The IDataset interface is used to handle a data source.
 */

/// @cond pluginAPI

/**
 * @note When developing a plug-in consider using the data format engine's implementation. 
 * You should <b>redefine this interface implementation</b>, only in case of <b>very specific needs.</b>
 * <p>
 */

/// @endcond pluginAPI

/**
 * A IDataset is a physical storage of CDMA objects, it holds a reference
 * of a root group, which is the root of a tree of IGroup and IDataItem. It is
 * the entry point to access the data structure it represents.<br>
 * For example in case of a data file container, the IDataset should refer to
 * the file handle.
 * 
 * @author nxi
 */
public interface IDataset extends IModelObject {

    /**
     * Close the dataset.
     * 
     * @throws IOException
     */
    void close() throws IOException;

    /**
     * Return the root group of the dataset.
     * 
     * @return IGroup that is on top of the structure 
     */
    IGroup getRootGroup();

    /**
     * Return the logical root of the dataset.
     * 
     * @return ILogicalGroup that is on top of the logical structure 
     */
    LogicalGroup getLogicalRoot();

    /**
     * Return the location of the dataset. If it's a file it will return the path.
     * 
     * @return String type 
     */
    String getLocation();

    /**
     * Return the title of the dataset.
     * 
     * @return string title  
     */
    String getTitle();

    /**
     * Set the location of the dataset.
     * 
     * @param location in String type 
     */
    void setLocation(String location);

    /**
     * Set the title for the Dataset.
     * 
     * @param title a String object 
     */
    void setTitle(String title);

    /**
     * Synchronize the dataset with the file reference.
     * 
     * @return true or false
     * @throws IOException
     */
    boolean sync() throws IOException;

    /**
     * Open the dataset. If it is a file should open the file,
     * if a database enable connection, etc.
     * 
     * @throws IOException
     */
    void open() throws IOException;

    /**
     * Save the contents / changes of the dataset to the file.
     * 
     * @throws WriterException
     *             failed to write 
     */
    void save() throws WriterException;

    /**
     * Save the contents of the dataset to a new location.
     * 
     * @throws WriterException
     *             failed to write 
     */
    void saveTo(String location) throws WriterException;

    /**
     * Save the specific contents / changes of the dataset.
     * 
     * @throws WriterException failed to write 
     */
    void save(IContainer container) throws WriterException;

    /**
     * Save the attribute to the specific path of the file.
     * 
     * @throws WriterException failed to write 
     */  
    void save(String parentPath, IAttribute attribute) throws WriterException;

    /**
     * Check if the data set is open.
     * 
     * @return true or false
     */
    boolean isOpen();
    
    
    /**
     * Return the last modification of the dataset.
     * 
     * @return long representing the last modification (in milliseconds)
     */
    long getLastModificationDate();
}
