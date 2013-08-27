package org.cdma.engine.hdf.navigation;

import java.io.File;
import java.io.IOException;

import javax.swing.tree.DefaultMutableTreeNode;

import ncsa.hdf.hdf5lib.H5;
import ncsa.hdf.object.FileFormat;
import ncsa.hdf.object.h5.H5File;
import ncsa.hdf.object.h5.H5Group;

import org.cdma.Factory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;

public class HdfDataset implements IDataset, Cloneable {

    private final String factoryName;
    private File hdfFileName;
    private H5File h5File;
    private String title;
    private IGroup root;

    public HdfDataset(String factoryName, File hdfFile) {
        this.factoryName = factoryName;
        this.hdfFileName = hdfFile;
        this.title = hdfFile.getName();
        this.h5File = new H5File(hdfFileName.getAbsolutePath(), H5File.WRITE);
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
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
                DefaultMutableTreeNode theRoot = (DefaultMutableTreeNode) h5File.getRootNode();
                if (theRoot != null) {
                    H5Group rootObject = (H5Group) theRoot.getUserObject();
                    root = new HdfGroup(factoryName, rootObject, null);
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
            root.save(this.h5File, root.getH5Group());
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

            // FileFormat fileToWrite = h5File.createFile(location, H5File.FILE_CREATE_DELETE);
            H5File fileToWrite = new H5File(location, FileFormat.CREATE);
            fileToWrite.open();

            HdfGroup root = (HdfGroup) getRootGroup();
            // recursive save
            root.save(fileToWrite, root.getH5Group());
            fileToWrite.close();

        } catch (Exception e) {
            // TODO DEBUG
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

    public static String readAviex(HdfDataset ds) {
        String result = null;
        String fileName = "/home/viguier/NeXusFiles/BackToMama/datas/bigtree.nxs";

        try {
            // File file = new File(fileName);
            //
            // HdfDataset ds = new HdfDataset("HDF", file);
            // ds.open();
            IGroup root = ds.getRootGroup();

            IGroup entry = root.getGroup("BSA_0006");
            IGroup swingGroup = entry.getGroup("SWING");
            IGroup aviexGroup = swingGroup.getGroup("Aviex");
            IDataItem item = aviexGroup.getDataItem("type");

            Object data = item.getData().getStorage();

            result = ((String[]) data)[0];
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    public static String readSampleInfos(HdfDataset ds) {
        String result = null;
        String fileName = "/home/viguier/NeXusFiles/BackToMama/BigTree/hdf.nxs";

        try {

            IGroup root = ds.getRootGroup();

            IGroup entry = root.getGroup("BSA_0006");
            IGroup sampleInfo = entry.getGroup("sample_info");
            IGroup comments = sampleInfo.getGroup("comments");
            IDataItem item = comments.getDataItem("data");

            // TODO DEBUG
            Object data = item.getData().getStorage();

            result = ((String[]) data)[0];
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    public static void main(String args[]) throws IOException, WriterException, NullPointerException {
        try {
            System.setProperty(H5.H5PATH_PROPERTY_KEY, "/home/viguier/LocalSoftware/hdf-java/lib/linux/libjhdf5.so");
            String fileName = "/home/viguier/NeXusFiles/BackToMama/datas/bigtree.nxs";

            File file = new File(fileName);

            final HdfDataset ds = new HdfDataset("HDF", file);
            ds.open();
            ds.saveTo("/home/viguier/out.nxs");
            // ds.saveWithNativeAPI("/home/viguier/out.nxs");
        } catch (Exception e) {
            e.printStackTrace();
        }
        //
        // Thread readAviexRunnable = new Thread() {
        // @Override
        // public void run() {
        // String result = null;
        // while (true) {
        // result = readAviex(ds);
        // System.out.println("Aviex Data Item = " + result);
        // }
        // }
        // };
        //
        // Thread readCommentRunnable = new Thread() {
        // @Override
        // public void run() {
        // String result = null;
        // while (true) {
        // result = readSampleInfos(ds);
        // System.out.println("Comments Item = " + result);
        // }
        // }
        // };
        //
        // readAviexRunnable.start();
        // readCommentRunnable.start();

    }
}
