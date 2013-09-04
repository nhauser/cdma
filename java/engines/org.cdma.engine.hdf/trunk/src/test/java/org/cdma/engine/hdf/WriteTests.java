package org.cdma.engine.hdf;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;

import org.cdma.engine.hdf.array.HdfArray;
import org.cdma.engine.hdf.navigation.HdfAttribute;
import org.cdma.engine.hdf.navigation.HdfDataItem;
import org.cdma.engine.hdf.navigation.HdfDataset;
import org.cdma.engine.hdf.navigation.HdfGroup;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class WriteTests {

    public static final File FIRST_FILE_TO_WRITE = new File("/home/viguier/testWriteFromScratch.nxs");
    public static final File SECOND_FILE_TO_WRITE = new File("/home/viguier/testCopyIntoNewFile.nxs");

    private static final String FACTORY_NAME = "HDF";

    private HdfArray createRandom1DArray(int arrayLength) {
        HdfArray result = null;

        double[] values = new double[arrayLength];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.random() * 1000;
        }
        int[] shape = { 1, arrayLength };

        try {
            result = new HdfArray(FACTORY_NAME, values, shape);
        } catch (InvalidArrayTypeException e) {
            e.printStackTrace();
        }
        return result;
    }

    @Test
    public void dTestModifyInExistingFile() throws InvalidArrayTypeException, WriterException, IOException {
        System.out.println("--------------------------------------------------");
        System.out.println("Test: Modify existing file");
        System.out.println(" - Modify group1/data1 values to [0,1,2,3]");
        HdfDataset dataset = new HdfDataset(FACTORY_NAME, FIRST_FILE_TO_WRITE, true);

        HdfGroup root = (HdfGroup) dataset.getRootGroup();

        HdfGroup group1 = (HdfGroup)root.getGroup("group1");

        // Modifiy dataItem values
        IDataItem dataItem = group1.getDataItem("data1");

        int[] values = { 0, 1, 2, 3 };
        int[] shape = { 1, 4 };
        IArray newArray = new HdfArray(FACTORY_NAME, values, shape);
        dataItem.setCachedData(newArray, false);

        assertEquals(int.class, newArray.getElementType());
        assertArrayEquals(shape, newArray.getShape());

        // Modify group name
        HdfGroup group2 = (HdfGroup) root.getGroup("group2");
        group2.setShortName("group2Modified");
        System.out.println(" - Rename group2 to group2Modified");

        dataset.save();
        dataset.close();
        System.out.println("End of test: Modify existing file");
        System.out.println("--------------------------------------------------");
    }

    @Test
    public void cTestWriteIntoNewFile() throws InvalidArrayTypeException, WriterException, IOException {
        System.out.println("--------------------------------------------------");
        System.out.println("Test: Copy existing dataset into new file and add new group & dataitem");
        if (SECOND_FILE_TO_WRITE.exists()) {
            if (!SECOND_FILE_TO_WRITE.delete()) {
                System.out.println("Cannot delete file: missing close() ??");
            }
        }

        HdfDataset dataset = new HdfDataset(FACTORY_NAME, FIRST_FILE_TO_WRITE);

        HdfGroup root = (HdfGroup) dataset.getRootGroup();
        IGroup group3 = new HdfGroup(FACTORY_NAME, "group3", "/group3", root);

        HdfDataItem dataItem = new HdfDataItem(FACTORY_NAME, "data3");

        HdfArray array = createRandom1DArray(10);
        dataItem.setCachedData(array, false);
        group3.addDataItem(dataItem);
        root.addSubgroup(group3);

        assertEquals(3, root.getGroupList().size());

        dataset.saveTo(SECOND_FILE_TO_WRITE.getAbsolutePath());

        dataset.close();
        System.out.println("End of test: Copy existing dataset into new file");
        System.out.println("--------------------------------------------------");
    }

    @Test
    public void bTestWriteIntoExistingFile() throws InvalidArrayTypeException, WriterException, IOException {
        System.out.println("--------------------------------------------------");
        System.out.println("Test: Write into a the previous file");

        HdfDataset dataset = new HdfDataset(FACTORY_NAME, FIRST_FILE_TO_WRITE, true);

        HdfGroup root = (HdfGroup) dataset.getRootGroup();
        IGroup group2 = new HdfGroup(FACTORY_NAME, "group2", "/group2", root);

        HdfDataItem dataItem = new HdfDataItem(FACTORY_NAME, "data2");
        dataItem.addStringAttribute("attr1", "mon attribut");
        dataItem.addOneAttribute(new HdfAttribute(FACTORY_NAME, "attr2", 5));

        HdfArray array = createRandom1DArray(10);
        dataItem.setCachedData(array, false);
        group2.addDataItem(dataItem);
        root.addSubgroup(group2);

        assertEquals(2, root.getGroupList().size());

        dataset.save();
        dataset.close();
        System.out.println("End of test: Write into the previous file");
        System.out.println("--------------------------------------------------");
    }

    @Test
    public void aTestWriteFromScratch() throws InvalidArrayTypeException, WriterException, IOException {
        System.out.println("--------------------------------------------------");
        System.out.println("Test: Write into a new file");

        if (FIRST_FILE_TO_WRITE.exists()) {
            if (!FIRST_FILE_TO_WRITE.delete()) {
                System.out.println("Cannot delete file: missing close() ??");
            }
        }
        HdfDataset dataset = new HdfDataset(FACTORY_NAME, FIRST_FILE_TO_WRITE);

        // Test Root Group
        HdfGroup root = (HdfGroup) dataset.getRootGroup();
        assertNotNull(root);
        assertTrue(root.isRoot());

        // Test Sub Group
        IGroup group = new HdfGroup(FACTORY_NAME, "group1", "/group1", root);
        root.addSubgroup(group);
        group.addStringAttribute("attr1", "mon attribut");
        group.addOneAttribute(new HdfAttribute(FACTORY_NAME, "attr2", 5));
        assertEquals("Attribute List size", 2, group.getAttributeList().size());
        assertEquals("group1", group.getShortName());
        assertEquals("/group1", group.getName());
        assertEquals(root, group.getRootGroup());
        assertEquals(dataset, group.getDataset());
        assertTrue(group.isEntry());
        assertFalse(group.isRoot());


        // Test Data Item
        HdfDataItem dataItem = new HdfDataItem(FACTORY_NAME, "data1");
        group.addDataItem(dataItem);
        HdfArray array = createRandom1DArray(10);
        dataItem.setCachedData(array, false);
        assertEquals(double.class, array.getElementType());
        assertEquals(root, dataItem.getRootGroup());
        assertEquals(group, dataItem.getParentGroup());
        assertEquals(dataset, dataItem.getDataset());
        assertEquals("/group1/data1", dataItem.getName());
        assertEquals("data1", dataItem.getShortName());

        dataset.save();
        dataset.close();
        System.out.println("End of test: Write into a new file");
        System.out.println("--------------------------------------------------");
    }
}
