package org.cdma.plugin.soleil.nexus;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;

import org.cdma.engine.hdf.navigation.HdfAttribute;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.nexus.array.NxsArray;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataItem;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataset;
import org.cdma.plugin.soleil.nexus.navigation.NxsGroup;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class WriteTests {

    public static final File FIRST_FILE_TO_WRITE = new File("/home/viguier/testNxsWriteFromScratch.nxs");
    public static final File SECOND_FILE_TO_WRITE = new File("/home/viguier/testNxsCopyIntoNewFile.nxs");
    private static int test_index = 1;

    private static final String FACTORY_NAME = "Nxs";

    private NxsArray createRandom1DArray(int arrayLength) {
        NxsArray result = null;

        double[] values = new double[arrayLength];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.random() * 1000;
        }
        int[] shape = { 1, arrayLength };

        try {
            result = new NxsArray(values, shape);
        } catch (InvalidArrayTypeException e) {
            e.printStackTrace();
        }
        return result;
    }

    @Test
    public void dTestModifyInExistingFile() throws InvalidArrayTypeException, WriterException, IOException,
    NoResultException {
        System.out.println("----------------------" + test_index++ + "----------------------------");
        System.out.println("Test: Modify existing file");
        System.out.println(" - Modify group1/data1 values to [0,1,2,3]");
        NxsDataset dataset = NxsDataset.instanciate(FIRST_FILE_TO_WRITE.toURI(), true);

        NxsGroup root = (NxsGroup) dataset.getRootGroup();
        assertNotNull(root);

        NxsGroup group1 = (NxsGroup) root.getGroup("group1");
        assertNotNull(group1);

        // Modifiy dataItem values
        IDataItem dataItem = group1.getDataItem("data1");
        assertNotNull(dataItem);

        int[] values = { 0, 1, 2, 3 };
        int[] shape = { 1, 4 };
        IArray newArray = new NxsArray(values, shape);
        dataItem.setCachedData(newArray, false);

        assertEquals(int.class, newArray.getElementType());
        assertArrayEquals(shape, newArray.getShape());

        // Modify group name
        NxsGroup group2 = (NxsGroup) root.getGroup("group2");
        assertNotNull(group2);
        group2.setShortName("group2Modified");
        System.out.println(" - Rename group2 to group2Modified");

        dataset.save();
        dataset.close();
        System.out.println("End of test: Modify existing file");
        System.out.println("--------------------------------------------------");
    }

    @Test
    public void cTestWriteIntoNewFile() throws InvalidArrayTypeException, WriterException, IOException,
    NoResultException {
        System.out.println("----------------------" + test_index++ + "----------------------------");
        System.out.println("Test: Copy existing dataset into new file and add new group & dataitem");
        if (SECOND_FILE_TO_WRITE.exists()) {
            if (!SECOND_FILE_TO_WRITE.delete()) {
                System.out.println("Cannot delete file: missing close() ??");
            }
        }

        NxsDataset dataset = NxsDataset.instanciate(FIRST_FILE_TO_WRITE.toURI());

        NxsGroup root = (NxsGroup) dataset.getRootGroup();
        assertNotNull(root);
        IGroup group3 = new NxsGroup(dataset, "group3", "/", root);
        NxsDataItem dataItem = new NxsDataItem("data3", dataset);
        NxsArray array = createRandom1DArray(10);
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
    public void bTestWriteIntoExistingFile() throws InvalidArrayTypeException, WriterException, IOException,
    NoResultException {
        System.out.println("----------------------" + test_index++ + "----------------------------");
        System.out.println("Test: Write into a the previous file");

        NxsDataset dataset = NxsDataset.instanciate(FIRST_FILE_TO_WRITE.toURI(), true);
        dataset.open();
        NxsGroup root = (NxsGroup) dataset.getRootGroup();
        IGroup group2 = new NxsGroup(dataset, "group2", "/", root);

        NxsDataItem dataItem = new NxsDataItem("data2", dataset);
        dataItem.addStringAttribute("attr1", "mon attribut");
        dataItem.addOneAttribute(new HdfAttribute(FACTORY_NAME, "attr2", 5));

        NxsArray array = createRandom1DArray(10);
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
    public void aTestWriteFromScratch() throws InvalidArrayTypeException, WriterException, IOException,
    NoResultException {
        System.out.println("----------------------" + test_index++ + "----------------------------");
        System.out.println("Test: Write into a new file");

        if (FIRST_FILE_TO_WRITE.exists()) {
            if (!FIRST_FILE_TO_WRITE.delete()) {
                System.out.println("Cannot delete file: missing close() ??");
            }
        }
        NxsDataset dataset = NxsDataset.instanciate(FIRST_FILE_TO_WRITE.toURI());

        // Test Root Group
        NxsGroup root = (NxsGroup) dataset.getRootGroup();
        assertNotNull(root);
        assertTrue(root.isRoot());

        // Test Sub Group
        IGroup group = new NxsGroup(dataset, "group1", "/", root);
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
        NxsDataItem dataItem = new NxsDataItem("data1", dataset);
        group.addDataItem(dataItem);
        NxsArray array = createRandom1DArray(10);
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
