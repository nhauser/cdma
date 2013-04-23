//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.nexus.navigation;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.nexusformat.NexusException;
import org.nexusformat.NexusFile;

import fr.soleil.nexus.DataItem;
import fr.soleil.nexus.NexusFileHandler;
import fr.soleil.nexus.NexusFileWriter;
import fr.soleil.nexus.NexusNode;
import fr.soleil.nexus.PathData;
import fr.soleil.nexus.PathGroup;
import fr.soleil.nexus.PathNexus;

public abstract class NexusDataset implements IDataset, Cloneable {
	
	static public boolean checkNeXusAPI() {
		// Check library can be fully loaded (specially the native part)
		boolean result = false;
    	try {
    		NexusFileHandler.loadAPI();
            result = true;
		} catch (Exception exception) {
			// Nothing to be done API hasn't been found
		} catch( Error error ) {
			Factory.getLogger().log(Level.SEVERE, error.getMessage() );
			// Nothing to be done but API isn't valid
		}
		return result;
	}
	
    public NexusDataset(String factoryName, File nexusFile, int bufferSize, boolean resetBuffer) throws FileAccessException {
        mFactory      = factoryName;
        mRootPhysical = null;
        mN4TWriter    = new NexusFileWriter(nexusFile.getAbsolutePath());
        mN4TCurPath   = PathNexus.ROOT_PATH.clone();
        mTitle        = nexusFile.getName();
        mN4TCurPath.setFile(nexusFile.getAbsolutePath());
        mN4TWriter.setBufferSize(bufferSize);
        mN4TWriter.isSingleRawResult(true);
        mN4TWriter.setCompressedData(true);
        if( resetBuffer ) {
            mN4TWriter.resetBuffer();
        }
    }

    public NexusDataset(String factoryName, File nexusFile, boolean resetBuffer) throws FileAccessException {
        this(factoryName, nexusFile, N4T_BUFF_SIZE, resetBuffer);
    }

    public NexusDataset(NexusDataset dataset) {
        mFactory      = dataset.mFactory;
        mN4TWriter    = dataset.mN4TWriter;
        mN4TCurPath   = dataset.mN4TCurPath.clone();
        mTitle        = dataset.mTitle;
        mRootPhysical = dataset.mRootPhysical;
        mN4TWriter.isSingleRawResult(true);
    }

    @Override
    public void open() throws IOException {
    }

    /// Methods
    @Override
    public void close() throws IOException {
    }

    @Override
    public String getLocation() {
        if (mN4TCurPath != null) {
            return mN4TCurPath.getFilePath();
        }
        return null;
    }

    @Override
    public String getTitle() {
        return mTitle;
    }

    @Override
    public void save() throws WriterException {
        List<IDataItem> items = new ArrayList<IDataItem>();
        NexusGroup.getDescendentDataItem(items, mRootPhysical);
        try {
            // Save each IDataItem
            DataItem data;
            for (IDataItem item : items) {
                data = ((NexusDataItem) item).getN4TDataItem();
                mN4TWriter.writeData(data, data.getPath());
            }
        } catch (NexusException e) {
            throw new WriterException(e.getMessage(), e);
        }
    }

    @Override
    public void setLocation(String location) {
        String sCurFile = "";
        PathNexus path = new PathGroup(location.split("/"));

        if (!mRootPhysical.equals(PathNexus.ROOT_PATH)) {
            sCurFile = mRootPhysical.getLocation();
        }

        try {
            mN4TWriter.openPath(path);
        } catch (NexusException e1) {
            NexusNode topNode = path.popNode();
            path = new PathData((PathGroup) path, topNode.getNodeName());
            try {
                mN4TWriter.openPath(path);
            } catch (NexusException e2) {
            }
        }
        mN4TCurPath = path;
        mN4TCurPath.setFile(sCurFile);
    }

    @Override
    public void setTitle(String title) {
        mTitle = title;
    }

    @Override
    public boolean isOpen() {
        return mN4TWriter.isFileOpened();
    }

    public NexusFileWriter getHandler() {
        return mN4TWriter;
    }
    
    @Override
    public long getLastModificationDate() {
        return mN4TWriter.getLastModificationDate();
    }
    
    @Override
    public String getFactoryName() {
        return mFactory;
    }

    @Override
    public IGroup getRootGroup() {
        if (mRootPhysical == null) {
            mRootPhysical = new NexusGroup(mFactory, null, PathNexus.ROOT_PATH.clone(), this);
        }
        return mRootPhysical;
    }

    @Override
    public boolean sync() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public void saveTo(String location) throws WriterException {
        throw new NotImplementedException();
    }

    @Override
    public void save(IContainer container) throws WriterException {
        throw new NotImplementedException();
    }

    @Override
    public void save(String parentPath, IAttribute attribute) throws WriterException {
        throw new NotImplementedException();
    }

    /**
     * Reset the internal buffer so the NexusDataset can see changes
     */
    public void resetBuffer() {
        mN4TWriter.resetBuffer();
    }
    
    // -----------------------------------------------------------
    /// protected methods
    // -----------------------------------------------------------
    // Accessors
    protected PathNexus getCurrentPath() {
        return mN4TCurPath;
    }

    protected PathNexus getRootPath() {
        return ((NexusGroup) mRootPhysical).getPathNexus();
    }

    protected void setLocation(PathNexus location) {
        try {
            mN4TWriter.openPath(location);
        } catch (NexusException e) {
        }
        mN4TCurPath = location.clone();
    }

    // Methods
    protected void setRootGroup(PathNexus rootPath) {
        mRootPhysical = new NexusGroup(mFactory, rootPath, this);
    }

    private IGroup           mRootPhysical;      // Physical root of the document
    private NexusFileWriter  mN4TWriter;         // Instance manipulating the NeXus file
    private String           mTitle;
    private String           mFactory;
    private PathNexus        mN4TCurPath;         // Instance of the current path
    private static final int N4T_BUFF_SIZE = 300; // Size of the buffer managed by NeXus4Tango

}
