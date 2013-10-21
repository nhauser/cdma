package org.cdma.plugin.soleil.edf.navigation;

import java.io.File;
import java.io.IOException;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.edf.EdfFactory;
import org.cdma.plugin.soleil.edf.dictionary.EdfLogicalGroup;

public class EdfDataset implements IDataset, Cloneable {

    private String filePath;
    private File file;
    private EdfGroup rootGroup;
    private EdfLogicalGroup logicalRoot;
    private boolean open = false;;

    public EdfDataset(String filePath) {
        super();
        setLocation(filePath);
        try {
            open();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    @Override
    public IDataset clone() {
        return null;
    }

    @Override
    public synchronized void close() throws IOException {
        open = false;
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
        throw new NotImplementedException();
    }

    @Override
    public synchronized void setLocation(String location) {
        this.filePath = location;
        this.file = null;
        this.rootGroup = null;
    }

    @Override
    public void setTitle(String title) {
        throw new NotImplementedException();
    }

    @Override
    public boolean sync() throws IOException {
        return false;
    }

    @Override
    public void open() throws IOException {
        if (!open) {
            if ((filePath == null) || (filePath.trim().isEmpty())) {
                file = null;
                rootGroup = null;
                throw new IOException("No root directory defined");
            } else {
                if (file == null || rootGroup == null) {
                    file = new File(filePath);
                    rootGroup = new EdfGroup(this, file);
                    // rootGroup.addSubgroup(new EdfGroup(file));
                    this.open = true;
                }
            }
        }
    }

    @Override
    public void save() {
        throw new NotImplementedException();
    }

    @Override
    public synchronized boolean isOpen() {
        return open;
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
