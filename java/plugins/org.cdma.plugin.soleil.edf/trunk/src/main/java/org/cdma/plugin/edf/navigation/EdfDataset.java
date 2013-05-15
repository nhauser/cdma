package org.cdma.plugin.edf.navigation;

import java.io.File;
import java.io.IOException;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.edf.EdfFactory;
import org.cdma.plugin.edf.dictionary.EdfLogicalGroup;

public class EdfDataset implements IDataset, Cloneable {

    private String directoryPath;
    private File directory;
    private EdfGroup rootGroup;
    private EdfLogicalGroup logicalRoot;

    public EdfDataset(String directoryPath) {
        super();
        setLocation(directoryPath);
    }

    @Override
    public IDataset clone() {
        return null;
    }

    @Override
    public synchronized void close() throws IOException {
        directory = null;
        rootGroup = null;
    }

    @Override
    public IGroup getRootGroup() {
        return rootGroup;
    }

    @Override
    public String getLocation() {
        return directoryPath;
    }

    @Override
    public String getTitle() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public synchronized void setLocation(String location) {
        this.directoryPath = location;
        this.directory = null;
        this.rootGroup = null;
    }

    @Override
    public void setTitle(String title) {
        // TODO Auto-generated method stub

    }

    @Override
    public boolean sync() throws IOException {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public void open() throws IOException {
        if ((directoryPath == null) || (directoryPath.trim().isEmpty())) {
            directory = null;
            rootGroup = null;
            throw new IOException("No root directory defined");
        }
        else {
            directory = new File(directoryPath);
            try {
                if (!directory.isDirectory()) {
                    directory = null;
                    rootGroup = null;
                    throw new IOException(directoryPath + " is not a directory");
                }
            }
            catch (SecurityException se) {
                directory = null;
                rootGroup = null;
                throw new IOException(se);
            }
            rootGroup = new EdfGroup(this, null);
            rootGroup.addSubgroup(new EdfGroup(directory));
        }
    }

    @Override
    public void save() {
        // TODO Auto-generated method stub

    }


    @Override
    public synchronized boolean isOpen() {
        return (directory != null);
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        if (logicalRoot == null) {
            logicalRoot = new EdfLogicalGroup(null, null, this);
        }
        return logicalRoot;
    }

    @Override
    public String getFactoryName() {
        return EdfFactory.NAME;
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
    public void save(String parentPath, IAttribute attribute) throws WriterException {
        // TODO Auto-generated method stub

    }

    @Override
    public long getLastModificationDate() {
        // TODO Auto-generated method stub
        return 0;
    }

}
