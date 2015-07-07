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
package org.cdma.plugin.soleil.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.navigation.NxsDataItem;
import org.cdma.plugin.soleil.navigation.NxsGroup;
import org.cdma.plugin.soleil.utils.NxsConstant;
import org.cdma.utils.Utilities.ModelType;

import fr.soleil.nexus.DataItem;
import fr.soleil.nexus.NexusNode;

/**
 * For each container in the context, this method will try to find its meta-data:
 * - acquisition_sequence (based on NXentry)
 * - region (of the detector it belongs to, based on number at end of the node's name)
 * 
 * @param context
 * @throws CDMAException
 */

public class HarvestSignalAttributes implements IPluginMethod {

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> inList = context.getContainers();
        List<IContainer> outList = new ArrayList<IContainer>();

        for (IContainer container : inList) {
            ModelType type = container.getModelType();
            outList.add(container);
            switch (type) {
                case Group: {
                    NxsGroup group = (NxsGroup) container;
                    NexusNode[] nodes = group.getPathNexus().getNodes();

                    setAttributeAcquisitionSequence(container, nodes);
                    break;
                }
                case DataItem: {
                    NxsDataItem item = (NxsDataItem) container;

                    // Try to set attributes
                    DataItem[] n4tItems = item.getNexusItems();
                    NexusNode[] nodes = n4tItems[0].getPath().getNodes();

                    // Set scan acquisition
                    setAttributeAcquisitionSequence(container, nodes);
                    setAttributeEquipment(container, nodes);
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

    private void setAttributeAcquisitionSequence(IContainer container, NexusNode[] nodes) {
        // Scan attribute
        if (nodes.length > 0) {
            NxsGroup root = (NxsGroup) container.getRootGroup();
            NexusNode[] rootNodes = root.getPathNexus().getNodes();
            if (rootNodes.length == 0 || !rootNodes[0].getClassName().equals("NXentry")) {
                root = (NxsGroup) root.getGroup(nodes[0].getNodeName());
            }
            String attrName = NxsConstant.ATTR_SCAN;
            String attrValue = root.getShortName();
            container.addStringAttribute(attrName, attrValue);
        }
    }

    private void setAttributeEquipment(IContainer container, NexusNode[] nodes) {
        // Scan attribute
        if (nodes.length > 1 && nodes[1].getClassName().equals("NXdata")) {

            String[] containerNameParts = container.getShortName().split("_");
            // By convention @SOLEIL, the equipment is before the '_'
            if (containerNameParts.length > 0) {
                container.addStringAttribute(NxsConstant.ATTR_EQUIPMENT, containerNameParts[0]);
            }

            NxsGroup root = (NxsGroup) container.getRootGroup();
            NexusNode[] rootNodes = root.getPathNexus().getNodes();
            if (rootNodes.length == 0 || !rootNodes[0].getClassName().equals("NXentry")) {
                root = (NxsGroup) root.getGroup(nodes[0].getNodeName());
            }

            String attrName = "region";
            String attrValue = nodes[nodes.length - 1].getNodeName().replaceAll(".*([0-9]+)$", "$1");
            container.addStringAttribute(attrName, attrValue);
        }
    }
}
