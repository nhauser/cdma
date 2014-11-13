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
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.cdma.engine.hdf.navigation.HdfDataset;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.junit.Test;

public class ReadTests {

    @Test
    public void aReadFile() throws Exception {
        IDataset dataSet = new HdfDataset("HDF", WriteTests.FIRST_FILE_TO_WRITE);
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
            int[] array = (int[]) storage;
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

            containers = root.findAllContainerByPath("/group*/data*");
            assertNotNull(containers);
            assertTrue(containers.size() == 2);


        }
    }

}
