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
package org.cdma.engine.hdf;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;

import ncsa.hdf.object.h5.H5ScalarDS;

import org.cdma.engine.hdf.array.HdfArray;
import org.cdma.engine.hdf.navigation.HdfAttribute;
import org.cdma.engine.hdf.navigation.HdfDataItem;
import org.cdma.engine.hdf.navigation.HdfDataset;
import org.cdma.engine.hdf.navigation.HdfGroup;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class WriteTests {

    public static final File FIRST_FILE_TO_WRITE = new File("C:/temp/testWriteFromScratch.nxs");
    public static final File SECOND_FILE_TO_WRITE = new File("C:/temp/testCopyIntoNewFile.nxs");

    private static final String FACTORY_NAME = "HDF";

    private HdfArray createRandom1DArray(final int arrayLength) {
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

    private double[] createImages(final int xarrayLength, final int yarrayLength, final int offset) {
        double[] values = new double[yarrayLength * xarrayLength];
        int index = 0;
        for (int i = 0; i < yarrayLength; i++) {
            for (int j = 0; j < xarrayLength; j++) {
                values[index] = index++ + offset;
            }
        }
        return values;
    }

    private double[][][] createImages2D(final int xarrayLength, final int yarrayLength, final int offset) {
        double[][][] values = new double[1][yarrayLength][xarrayLength];
        double[][] image = values[0];
        int index = 0;
        for (int i = 0; i < yarrayLength; i++) {
            for (int j = 0; j < xarrayLength; j++) {
                image[i][j] = index++ + offset;
            }
        }
        return values;
    }


    @Test
    public void dTestModifyInExistingFile() throws Exception {
        System.out.println("--------------------------------------------------");
        System.out.println("Test: Modify existing file");
        System.out.println(" - Modify group1/data1 values to [0,1,2,3]");
        HdfDataset dataset = new HdfDataset(FACTORY_NAME, FIRST_FILE_TO_WRITE, true);

        HdfGroup root = (HdfGroup) dataset.getRootGroup();

        HdfGroup group1 = (HdfGroup)root.getGroup("group1");

        // Modifiy dataItem values
        IDataItem dataItem = group1.getDataItem("data1");
        assertNotNull(dataItem);
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
    public void eTestWriteMultiIntoNewFile() throws Exception {
        System.out.println("--------------------------------------------------");
        System.out.println("Test: Copy existing dataset into new file and add new group & dataitem");
        if (FIRST_FILE_TO_WRITE.exists()) {
            if (!FIRST_FILE_TO_WRITE.delete()) {
                System.out.println("Cannot delete file: missing close() ??");
                System.exit(0);
            }
        }

        HdfDataset dataset = new HdfDataset(FACTORY_NAME, FIRST_FILE_TO_WRITE);

        // Create group with name group3 under root node and save it
        HdfGroup root = (HdfGroup) dataset.getRootGroup();
        IGroup group3 = new HdfGroup(FACTORY_NAME, "group3", "/", root, dataset);
        root.addSubgroup(group3);
        dataset.save();

        // Image are 20x10
        int xLength = 20;
        int yLength = 10;

        // We have 3 random images
        double[] image1 = createImages(xLength, yLength, 0);
        double[] image2 = createImages(xLength, yLength, 400);
        double[] image3 = createImages(xLength, yLength, 800);

        // So shape is:
        int[] shape = new int[] { 3, yLength, xLength };

        // We create a DataItem under group3
        HdfDataItem dataItem = new HdfDataItem(FACTORY_NAME, dataset.getH5File(), group3, "imageStack", shape,
                double.class);

        // Now we have to tune the underlying HDF item
        H5ScalarDS h5Item = dataItem.getH5DataItem();
        long[] selectedDims = h5Item.getSelectedDims();
        long[] startDims = h5Item.getStartDims();

        // We have to give to HDF the non-reduced shape of the slabs we are going to put in the dataitem
        selectedDims[0] = 1;
        selectedDims[1] = yLength;
        selectedDims[2] = xLength;

        // We have to modify the startDims because HDF cannot guess where to put the slab
        // First image is at index 0 on the first dimension of our 3 dimension hyperslab
        startDims[0] = 0; // optional, this is the default value
        dataItem.getH5DataItem().write(image1);

        // For the next image, we want start at index 1 on the first dimension of the hyperslab
        startDims[0] = 1;
        dataItem.getH5DataItem().write(image2);

        // For the next image, we want start at index 2 on the first dimension of the hyperslab
        startDims[0] = 2;
        dataItem.getH5DataItem().write(image3);

        dataset.save();
        dataset.close();
    }
    @Test
    public void cTestWriteIntoNewFile() throws Exception {
        System.out.println("--------------------------------------------------");
        System.out.println("Test: Copy existing dataset into new file and add new group & dataitem");
        if (SECOND_FILE_TO_WRITE.exists()) {
            if (!SECOND_FILE_TO_WRITE.delete()) {
                System.out.println("Cannot delete file: missing close() ??");
            }
        }

        HdfDataset dataset = new HdfDataset(FACTORY_NAME, FIRST_FILE_TO_WRITE);

        HdfGroup root = (HdfGroup) dataset.getRootGroup();
        IGroup group3 = new HdfGroup(FACTORY_NAME, "group3", "/", root, dataset);

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
    public void bTestWriteIntoExistingFile() throws Exception {
        System.out.println("--------------------------------------------------");
        System.out.println("Test: Write into a the previous file");

        // Create a dataset in _Append_ mode
        HdfDataset dataset = new HdfDataset(FACTORY_NAME, FIRST_FILE_TO_WRITE, true);

        HdfGroup root = (HdfGroup) dataset.getRootGroup();
        IGroup group2 = new HdfGroup(FACTORY_NAME, "group2", "/", root, dataset);

        HdfDataItem dataItem = new HdfDataItem(FACTORY_NAME, "data2");
        dataItem.addStringAttribute("attr1", "mon attribut");
        dataItem.addOneAttribute(new HdfAttribute(FACTORY_NAME, "attr2", 5));

        HdfArray array = createRandom1DArray(10);
        dataItem.setCachedData(array, false);
        group2.addDataItem(dataItem);
        root.addSubgroup(group2);

        assertEquals(2, root.getGroupList().size());

        // Test if we can acces to previously written group1
        HdfGroup group1 = (HdfGroup) root.getGroup("group1");
        assertNotNull(group1);
        group1.addStringAttribute("attr100", "mon attribut sauvé ensuite");
        HdfDataItem data1 = (HdfDataItem) group1.getDataItem("data1");
        assertNotNull(data1);
        data1.addStringAttribute("attr100", "mon attribut sauvé ensuite");


        dataset.save();
        dataset.close();
        System.out.println("End of test: Write into the previous file");
        System.out.println("--------------------------------------------------");
    }

    @Test
    public void aTestWriteFromScratch() throws Exception {
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
        IGroup group = new HdfGroup(FACTORY_NAME, "group1", "/", root, dataset);
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
