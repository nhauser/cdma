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
    private File hdfFileName;
    private H5File h5File;
    private String title;
    private IGroup root;

    public HdfDataset(String factoryName, File hdfFile) {
        this(factoryName, hdfFile, false);
    }

    public HdfDataset(String factoryName, File hdfFile, boolean appendToFile) {
        this.factoryName = factoryName;
        this.hdfFileName = hdfFile;
        this.title = hdfFile.getName();
        try {
            int flag = H5File.CREATE;
            if (hdfFile.exists()) {
                if (appendToFile) {
                    flag = H5File.WRITE;
                } else {
                    flag = H5File.READ;
                }
            }
            this.h5File = (H5File) new H5File(hdfFile.getAbsolutePath())
            .createInstance(hdfFile.getAbsolutePath(), flag);
        } catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public String getFactoryName() {
        return factoryName;
    }

    @Override
    public void close() throws IOException {
        try {
            h5File.close();
            h5File = null;
        } catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public IGroup getRootGroup() {
        if (root == null) {
            if (h5File != null) {
                try {
                    h5File.open();
                } catch (Exception e) {
                    e.printStackTrace();
                }
                DefaultMutableTreeNode theRoot = (DefaultMutableTreeNode) h5File.getRootNode();
                if (theRoot != null) {
                    H5Group rootObject = (H5Group) theRoot.getUserObject();
                    root = new HdfGroup(factoryName, rootObject, this);
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
    public void setLocation(String location) {
        hdfFileName = new File(location);
        try {
            open();
        } catch (IOException e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public String getLocation() {
        String result = null;
        if (hdfFileName != null) {
            result = hdfFileName.getAbsolutePath();
        }
        return result;
    }

    @Override
    public String getTitle() {
        return title;
    }

    @Override
    public void setTitle(String title) {
        this.title = title;
    }

    @Override
    public long getLastModificationDate() {
        return hdfFileName.lastModified();
    }

    @Override
    public void open() throws IOException {
        if (hdfFileName != null) {
            try {
                h5File.open();
            } catch (Exception e) {
                Factory.getLogger().severe(e.getMessage());
            }

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
            e.printStackTrace();
        }
    }

    @Override
    public void saveTo(String location) throws WriterException {
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
            e.printStackTrace();
            throw new WriterException(e);
        }
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
    public String toString() {
        StringBuffer buffer = new StringBuffer();
        buffer.append("Dataset = " + getLocation());
        return buffer.toString();
    }
}
