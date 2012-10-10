package org.cdma.plugin.soleil.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
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
                    setAttributeEquipment(container, nodes, outList);

                    break;
                }
                case DataItem: {
                    NxsDataItem item = (NxsDataItem) container;

                    // Try to set attributes
                    DataItem[] n4tItems = item.getNexusItems();
                    NexusNode[] nodes = n4tItems[0].getPath().getNodes();

                    // Set scan acquisition
                    setAttributeAcquisitionSequence(container, nodes);

                    // Node is under NXinstrument it belongs to an equipment
                    if (nodes.length > 2 && nodes[1].getClassName().equals("NXinstrument")) {
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

    private void setAttributeAcquisitionSequence(IContainer container, NexusNode[] nodes) {
        NxsGroup root = (NxsGroup) container.getRootGroup();
        
        // Scan attribute
        if (nodes.length > 0) {
            NexusNode[] rootNodes = root.getPathNexus().getNodes();
            if( rootNodes.length == 0 || ! rootNodes[0].getClassName().equals("NXentry") ) {
                root = (NxsGroup) root.getGroup(nodes[0].getNodeName());
            }
            String attrName = NxsConstant.ATTR_SCAN;
            String attrValue = root.getShortName();
            container.addStringAttribute(attrName, attrValue);
        }
    }

    private void setAttributeEquipment(IContainer container, NexusNode[] nodes, List<IContainer> outList) {
        // Set the root group at the NXentry position
        NxsGroup root = (NxsGroup) container.getRootGroup();
        NexusNode[] rootNodes = root.getPathNexus().getNodes();
        if( rootNodes.length == 0 || ! rootNodes[0].getClassName().equals("NXentry") ) {
            root = (NxsGroup) root.getGroup(nodes[0].getNodeName());
        }
        
        // Equipment attribute (NXdetector, NXmono...)
        if (container.getAttribute(NxsConstant.ATTR_EQUIPMENT) == null) {
            if (nodes.length > 2 && nodes[1].getClassName().equals("NXinstrument")) {
                String attrValue = root.getGroup(nodes[1].getNodeName())
                                       .getGroup(nodes[2].getNodeName())
                                       .getShortName();
                container.addStringAttribute(NxsConstant.ATTR_EQUIPMENT, attrValue);
            }
        }

        if (container.getAttribute(NxsConstant.ATTR_REGION) == null) {
            if (nodes.length > 2 && nodes[2].getClassName().equals("NXdetector")) {
                NexusNode node = nodes[2];
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
                    }
                    else {
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
                                }
                                else {
                                    IContainer clone = container.clone();
                                    clone.addStringAttribute(NxsConstant.ATTR_REGION, region);
                                    outList.add(clone);
                                }
                                first = false;
                            }
                        }

                    }
                }
                else {
                    container.addStringAttribute(NxsConstant.ATTR_REGION, "0");
                }
            }
        }
    }

}
