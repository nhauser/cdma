package org.cdma.plugin.edf.navigation;

import java.io.File;
import java.io.IOException;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.edf.EdfFactory;
import org.cdma.plugin.edf.dictionary.EdfLogicalGroup;

public class EdfDataset implements IDataset, Cloneable {

    private String filePath;
    private File file;
    private EdfGroup rootGroup;
    private EdfLogicalGroup logicalRoot;

    public EdfDataset(String filePath) {
        super();
        setLocation(filePath);
    }

    @Override
    public IDataset clone() {
        return null;
    }

    @Override
    public synchronized void close() throws IOException {
        file = null;
        rootGroup = null;
    }

    @Override
    public IGroup getRootGroup() {
        return rootGroup;
    }

    @Override
    public String getLocation() {
        return filePath;
    }

    @Override
    public String getTitle() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public synchronized void setLocation(String location) {
        this.filePath = location;
        this.file = null;
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
        if ((filePath == null) || (filePath.trim().isEmpty())) {
            file = null;
            rootGroup = null;
            throw new IOException("No root directory defined");
        }
        else {
            file = new File(filePath);
            try {
                if (file.isDirectory()) {
                    file = null;
                    rootGroup = null;
                    throw new IOException(filePath + " is not a directory");
                }
            }
            catch (SecurityException se) {
                file = null;
                rootGroup = null;
                throw new IOException(se);
            }
            rootGroup = new EdfGroup(this, file);
            // rootGroup.addSubgroup(new EdfGroup(file));
        }
    }

    @Override
    public void save() {
        // TODO Auto-generated method stub

    }


    @Override
    public synchronized boolean isOpen() {
        return (file != null);
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

    @Override
    public long getLastModificationDate() {
        return file.lastModified();
    }

}
