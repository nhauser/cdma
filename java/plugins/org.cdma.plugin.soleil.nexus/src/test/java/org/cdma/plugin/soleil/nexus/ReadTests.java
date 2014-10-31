/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.nexus;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ncsa.hdf.hdf5lib.H5;
import ncsa.hdf.hdf5lib.HDF5Constants;

import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.ISliceIterator;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataset;
import org.junit.Test;

public class ReadTests {

    @Test
    public void aReadFile() throws IOException, NoResultException {
        IDataset dataSet = NxsDataset.instanciate(WriteTests.FIRST_FILE_TO_WRITE.toURI());
        IGroup root = dataSet.getRootGroup();
        if (root != null) {

            // Test ROOT
            List<IGroup> groupList = root.getGroupList();
            assertEquals(2, groupList.size());
            assertNull(root.getParentGroup());

            // Test IGroup
            IGroup group1 = root.getGroup("group1");
            assertNotNull(group1);
            assertEquals(2, group1.getAttributeList().size());
            assertEquals("Attribute List size", 2, group1.getAttributeList().size());
            assertEquals("group1", group1.getShortName());
            assertEquals("/group1", group1.getName());
            assertEquals(root, group1.getRootGroup());
            assertEquals(dataSet, group1.getDataset());
            assertTrue(group1.isEntry());
            assertFalse(group1.isRoot());

            // Test IDataItem
            IDataItem data1 = group1.getDataItem("data1");
            assertNotNull(data1);
            IArray iArray = data1.getData();
            Object storage = iArray.getStorage();

            Object[] arrays = (Object[]) storage;
            int[] array = (int[]) arrays[0];
            int[] expected = { 0, 1, 2, 3 };
            assertArrayEquals(expected, array);
            assertEquals(int.class, iArray.getElementType());
            assertEquals(root, data1.getRootGroup());
            assertEquals(group1, data1.getParentGroup());
            assertEquals(dataSet, data1.getDataset());
            assertEquals("/group1/data1", data1.getName());
            assertEquals("data1", data1.getShortName());

            // Test Navigation
            IContainer container = root.findContainerByPath("/group1/data1");
            assertNotNull(container);

            List<IContainer> containers = root.findAllContainerByPath("/group1/data1");
            assertNotNull(containers);
            assertTrue(containers.size() == 1);

            containers = root.findAllContainerByPath("/group*Modified/data*");
            assertNotNull(containers);
            assertTrue(containers.size() == 1);

            containers = root.findAllContainerByPath("/group*/data*");
            assertNotNull(containers);
            assertTrue(containers.size() == 2);

        }
    }

    // Values for the status of space allocation
    enum H5Z_filter {
        H5Z_FILTER_ERROR(HDF5Constants.H5Z_FILTER_ERROR), H5Z_FILTER_NONE(HDF5Constants.H5Z_FILTER_NONE),
        H5Z_FILTER_DEFLATE(HDF5Constants.H5Z_FILTER_DEFLATE), H5Z_FILTER_SHUFFLE(HDF5Constants.H5Z_FILTER_SHUFFLE),
        H5Z_FILTER_FLETCHER32(HDF5Constants.H5Z_FILTER_FLETCHER32), H5Z_FILTER_SZIP(HDF5Constants.H5Z_FILTER_SZIP),
        H5Z_FILTER_NBIT(HDF5Constants.H5Z_FILTER_NBIT), H5Z_FILTER_SCALEOFFSET(HDF5Constants.H5Z_FILTER_SCALEOFFSET),
        H5Z_FILTER_RESERVED(HDF5Constants.H5Z_FILTER_RESERVED), H5Z_FILTER_MAX(HDF5Constants.H5Z_FILTER_MAX);
        private static final Map<Integer, H5Z_filter> lookup = new HashMap<Integer, H5Z_filter>();

        static {
            for (H5Z_filter s : EnumSet.allOf(H5Z_filter.class))
                lookup.put(s.getCode(), s);
        }

        private final int code;

        H5Z_filter(int layout_type) {
            this.code = layout_type;
        }

        public int getCode() {
            return this.code;
        }

        public static H5Z_filter get(int code) {
            return lookup.get(code);
        }
    }

    @Test
    public void bReadCompressedFile() {
        String FILENAME = "D:/Samples/Nanoscopium/series_269_master.h5";
        File fileToRead = new File(FILENAME);
        String DATASETNAME = "DS1";
        int DIM_X = 32;
        int DIM_Y = 64;
        int CHUNK_X = 4;
        int CHUNK_Y = 8;
        int RANK = 2;
        int NDIMS = 2;

        int file_id = -1;
        int dataset_id = -1;
        int dcpl_id = -1;
        int[][] dset_data = new int[DIM_X][DIM_Y];
        IDataset dataSet = null;

        // Open an existing file.
        try {
            dataSet = NxsDataset.instanciate(fileToRead.toURI());
            if (dataSet != null) {
                IGroup root = dataSet.getRootGroup();
                IGroup entry = root.getGroup("entry");
                if (entry != null) {
                    IDataItem data = entry.getDataItem("data_000000");
                    IArray array = data.getData();
                    for (ISliceIterator it = array.getSliceIterator(2); it.hasNext();) {
                        IArray slideArray = it.getArrayNext();
                        // TODO DEBUG
                        System.out.println(Arrays.toString(slideArray.getShape()));
                        Object dataArray = slideArray.getStorage();
                    }
                    // TODO DEBUG
                    System.out.println("STOP");
                }


            }
            //            file_id = H5.H5Fopen(FILENAME, HDF5Constants.H5F_ACC_RDONLY, HDF5Constants.H5P_DEFAULT);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Open an existing dataset.
        try {
            if (file_id >= 0)
                dataset_id = H5.H5Dopen(file_id, DATASETNAME, HDF5Constants.H5P_DEFAULT);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Retrieve the dataset creation property list.
        try {
            if (dataset_id >= 0)
                dcpl_id = H5.H5Dget_create_plist(dataset_id);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Retrieve and print the filter type. Here we only retrieve the
        // first filter because we know that we only added one filter.
        try {
            if (dcpl_id >= 0) {
                // Java lib requires a valid filter_name object and cd_values
                int[] flags = { 0 };
                long[] cd_nelmts = { 1 };
                int[] cd_values = { 0 };
                String[] filter_name = { "" };
                int[] filter_config = { 0 };
                int filter_type = -1;

                filter_type = H5
                        .H5Pget_filter(dcpl_id, 0, flags, cd_nelmts, cd_values, 120, filter_name, filter_config);
                System.out.print("Filter type is: ");
                switch (H5Z_filter.get(filter_type)) {
                    case H5Z_FILTER_DEFLATE:
                        System.out.println("H5Z_FILTER_DEFLATE");
                        break;
                    case H5Z_FILTER_SHUFFLE:
                        System.out.println("H5Z_FILTER_SHUFFLE");
                        break;
                    case H5Z_FILTER_FLETCHER32:
                        System.out.println("H5Z_FILTER_FLETCHER32");
                        break;
                    case H5Z_FILTER_SZIP:
                        System.out.println("H5Z_FILTER_SZIP");
                        break;
                    case H5Z_FILTER_NBIT:
                        System.out.println("H5Z_FILTER_NBIT");
                        break;
                    case H5Z_FILTER_SCALEOFFSET:
                        System.out.println("H5Z_FILTER_SCALEOFFSET");
                        break;
                    default:
                        System.out.println("H5Z_FILTER_ERROR");
                }
                System.out.println();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Read the data using the default properties.
        try {
            if (dataset_id >= 0) {
                H5.H5Dread(dataset_id, HDF5Constants.H5T_NATIVE_INT, HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
                        HDF5Constants.H5P_DEFAULT, dset_data);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
