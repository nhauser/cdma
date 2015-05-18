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
package org.cdma.engine.hdf.navigation;

import java.io.File;
import java.io.IOException;

import javax.swing.tree.DefaultMutableTreeNode;

import ncsa.hdf.object.FileFormat;
import ncsa.hdf.object.h5.H5File;
import ncsa.hdf.object.h5.H5Group;

import org.cdma.Factory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;

public class HdfDataset implements IDataset, Cloneable {

    private final String factoryName;
    private String hdfFileName;
    private H5File h5File;
    private String title;
    private IGroup root;
    private int openFlag;

    public HdfDataset(final String factoryName, final File hdfFile) throws Exception {
        this(factoryName, hdfFile, false);
    }

    public HdfDataset(final String factoryName, final File hdfFile, final boolean appendToFile) throws Exception {
        this.factoryName = factoryName;
        this.hdfFileName = hdfFile.getAbsolutePath();
        this.title = hdfFile.getName();
        try {
            openFlag = H5File.CREATE;
            if (hdfFile.exists()) {
                if (appendToFile) {
                    openFlag = H5File.WRITE;
                } else {
                    openFlag = H5File.READ;
                }
            }
            initHdfFile();
        } catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
            throw e;
        }
    }

    private void initHdfFile() throws Exception {
        if (hdfFileName != null) {
            this.h5File = (H5File) new H5File(hdfFileName).createInstance(hdfFileName, openFlag);
            this.h5File.open();
        }
    }

    @Override
    public String getFactoryName() {
        return factoryName;
    }


    @Override
    public IGroup getRootGroup() {
        if (root == null) {
            if (h5File != null) {
                try {
                    h5File.open();
                } catch (Exception e) {
                    e.printStackTrace();
                    return null;
                }
                DefaultMutableTreeNode theRoot = (DefaultMutableTreeNode) h5File.getRootNode();
                if (theRoot != null) {
                    H5Group rootObject = (H5Group) theRoot.getUserObject();
                    root = new HdfGroup(factoryName, rootObject, (HdfGroup) null, this);
                }
            }
        }
        return root;
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        return new LogicalGroup(null, this);
    }

    @Override
    public void setLocation(final String location) {
        hdfFileName = location;
        try {
            open();
        } catch (IOException e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public String getLocation() {
        return hdfFileName;
    }

    public H5File getH5File() {
        return this.h5File;
    }

    @Override
    public String getTitle() {
        return title;
    }

    @Override
    public void setTitle(final String title) {
        this.title = title;
    }

    @Override
    public long getLastModificationDate() {
        return new File(hdfFileName).lastModified();
    }

    @Override
    public void open() throws IOException {
        try{
            if (h5File != null) {
                this.h5File.open();
            }
        } catch (Exception e) {
            if (e instanceof IOException){
                IOException ioException = (IOException) e;
                throw ioException;
            }
        }
    }

    @Override
    public void close() throws IOException {
        try {
            h5File.close();
        } catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public boolean isOpen() {
        return h5File != null;
    }

    @Override
    public boolean sync() throws IOException {
        throw new NotImplementedException();
    }

    public static boolean checkHdfAPI() {
        return true;
    }

    @Override
    public void save() throws WriterException {

        HdfGroup root = (HdfGroup) getRootGroup();
        // recursive save
        try {
            root.save(this.h5File, null);
        } catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public void saveTo(final String location) throws WriterException {
        try {
            File newFile = new File(location);
            if (newFile.exists()) {
                newFile.delete();
            }

            H5File fileToWrite = new H5File(location, FileFormat.CREATE);
            fileToWrite.open();

            HdfGroup root = (HdfGroup) getRootGroup();
            // recursive save
            root.save(fileToWrite, null);
            fileToWrite.close();

        } catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
            throw new WriterException(e);
        }
    }

    @Override
    public void save(final IContainer container) throws WriterException {
        throw new NotImplementedException();
    }

    @Override
    public void save(final String parentPath, final IAttribute attribute) throws WriterException {
        throw new NotImplementedException();
    }

    @Override
    public String toString() {
        StringBuffer buffer = new StringBuffer();
        buffer.append("Dataset = " + getLocation());
        return buffer.toString();
    }
}
