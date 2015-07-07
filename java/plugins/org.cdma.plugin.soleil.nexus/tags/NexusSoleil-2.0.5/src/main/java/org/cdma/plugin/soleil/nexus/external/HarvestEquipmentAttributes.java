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
package org.cdma.plugin.soleil.nexus.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.nexus.NxsFactory;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataItem;
import org.cdma.plugin.soleil.nexus.navigation.NxsGroup;
import org.cdma.plugin.soleil.nexus.utils.NxsConstant;
import org.cdma.plugin.soleil.nexus.utils.NxsNode;
import org.cdma.utils.Utilities.ModelType;

/**
 * For each container in the context, this method will try to find its meta-data:
 * - acquisition_sequence (based on NXentry)
 * - region (of the detector it belongs to, based on number at end of the node's name)
 * - equipment (based on the name of NX... direct son of the NXinstrument)
 * 
 * @param context
 * @throws CDMAException
 */
public class HarvestEquipmentAttributes implements IPluginMethod {

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public void execute(final Context context) throws CDMAException {
        List<IContainer> inList = context.getContainers();
        List<IContainer> outList = new ArrayList<IContainer>();

        for (IContainer container : inList) {
            ModelType type = container.getModelType();
            outList.add(container);
            switch (type) {
                case Group: {
                    NxsGroup group = (NxsGroup) container;
                    NxsNode[] nodes = group.getNxsPath().getNodes();

                    setAttributeAcquisitionSequence(container, nodes);
                    setAttributeEquipment(container, nodes, outList);

                    break;
                }
                case DataItem: {
                    NxsDataItem item = (NxsDataItem) container;

                    NxsNode[] nodes = item.getNxsPath().getNodes();

                    // Set scan acquisition
                    setAttributeAcquisitionSequence(container, nodes);

                    if (nodes.length >= 2) {
                        setAttributeEquipment(container, nodes, outList);
                    }

                    break;
                }
                case LogicalGroup: {
                    break;
                }
                default: {
                    break;
                }
            }

        }
        context.setContainers(outList);
    }

    private void setAttributeAcquisitionSequence(final IContainer container, final NxsNode[] nodes) {
        NxsGroup root = (NxsGroup) container.getRootGroup();

        // Scan attribute
        if (nodes.length > 0) {
            // Check the root group is the real root of the file
            NxsNode current = (NxsNode) root.getNxsPath().getCurrentNode();
            if (current == null || !current.getClassName().equals("NXentry")) {
                // LogicalMode: root = NXentry = acquisition_sequence
                root = (NxsGroup) root.getGroup("<NXentry>");
            }
            /*
                        NxsNode[] rootNodes = root.getNxsPath().getNodes();
                        if( rootNodes.length == 0 || ! rootNodes[0].getClassName().equals("NXentry") ) {
                            root = (NxsGroup) root.getGroup(nodes[0].getNodeName());
                        }
             */
            String attrName = NxsConstant.ATTR_SCAN;
            String attrValue = root.getShortName();
            container.addStringAttribute(attrName, attrValue);
        }
    }

    private void setAttributeEquipment(final IContainer container, final NxsNode[] nodes, final List<IContainer> outList) {
        // Set the root group at the NXentry position
        NxsGroup root = (NxsGroup) container.getRootGroup();
        NxsNode[] rootNodes = root.getNxsPath().getNodes();
        if (rootNodes.length == 0 || !rootNodes[0].getClassName().equals("NXentry")) {
            root = (NxsGroup) root.getGroup(nodes[1].getNodeName());
        }

        // Equipment attribute (NXdetector, NXmono...)
        if (container.getAttribute(NxsConstant.ATTR_EQUIPMENT) == null) {
            if (nodes.length > 2 && nodes[2].getClassName().equals("NXinstrument")) {
                // String attrValue = container.getParentGroup().getShortName();
                String node2Name = nodes[2].getNodeName();
                String node3Name = nodes[3].getNodeName();
                IGroup group2 = root.getGroup(node2Name);
                IGroup group3 = group2.getGroup(node3Name);
                String attrValue = group3.getShortName();

                container.addStringAttribute(NxsConstant.ATTR_EQUIPMENT, attrValue);
            }
        }

        if (container.getAttribute(NxsConstant.ATTR_REGION) == null) {
            if (nodes.length > 2 && nodes[3].getClassName().equals("NXdetector")) {
                NxsNode node = nodes[2];
                // If Scienta
                if (node.getNodeName().toLowerCase().matches(".*scienta.*")) {
                    String region = node.getNodeName().replaceAll(".*([0-9]+)$", "$1");
                    container.addStringAttribute(NxsConstant.ATTR_REGION, region);
                }
                // If Xia
                else if (node.getNodeName().toLowerCase().matches("xia")) {
                    // We are on sub-node of the XIA the region is contained in the name
                    if (nodes.length >= 4) {
                        node = nodes[3];
                        String region = node.getNodeName().replaceAll(".*([0-9]+)$", "$1");
                        if (node.getNodeName().toLowerCase().matches(".*xia.*")) {
                            container.addStringAttribute(NxsConstant.ATTR_REGION, region);
                        }
                    } else {
                        // We are on the XIA the region is contained in the children name
                        IGroup xia = root.getGroup(nodes[1].getNodeName()).getGroup(nodes[2].getNodeName());
                        List<IDataItem> list = xia.getDataItemList();
                        String region;
                        List<String> regions = new ArrayList<String>();
                        boolean first = true;
                        for (IDataItem item : list) {
                            region = item.getName().replaceAll(".*([0-9]+)$", "$1");
                            if (!regions.contains(region)) {
                                regions.add(region);

                                if (first) {
                                    container.addStringAttribute(NxsConstant.ATTR_REGION, region);
                                } else {
                                    IContainer clone = container.clone();
                                    clone.addStringAttribute(NxsConstant.ATTR_REGION, region);
                                    outList.add(clone);
                                }
                                first = false;
                            }
                        }

                    }
                } else {
                    container.addStringAttribute(NxsConstant.ATTR_REGION, "0");
                }
            }
        }
    }

}
