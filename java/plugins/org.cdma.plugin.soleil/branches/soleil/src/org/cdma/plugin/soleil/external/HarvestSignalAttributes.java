package org.cdma.plugin.soleil.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.navigation.NxsDataItem;
import org.cdma.plugin.soleil.navigation.NxsGroup;
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
                    IGroup root = container.getDataset().getRootGroup();
                    NexusNode[] nodes = group.getPathNexus().getNodes();

                    setAttributeAcquisitionSequence(container, nodes, root);
                    break;
                }
                case DataItem: {
                    NxsDataItem item = (NxsDataItem) container;
                    IGroup root = container.getDataset().getRootGroup();

                    // Try to set attributes
                    DataItem[] n4tItems = item.getNexusItems();
                    NexusNode[] nodes = n4tItems[0].getPath().getNodes();

                    // Set scan acquisition
                    setAttributeAcquisitionSequence(container, nodes, root);
                    setAttributeEquipment(container, nodes, root);
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

    private void setAttributeAcquisitionSequence(IContainer container, NexusNode[] nodes, IGroup root) {
        // Scan attribute
        if (nodes.length > 0 && nodes[0].getClassName().equals("NXentry")) {
            String attrName = "acquisition_sequence";
            String attrValue = root.getShortName();
            container.addStringAttribute(attrName, attrValue);
        }
    }
    
    private void setAttributeEquipment(IContainer container, NexusNode[] nodes, IGroup root) {
        // Scan attribute
        if (nodes.length > 1 && nodes[1].getClassName().equals("NXdata")) {
            String attrName = "region";
            String attrValue = nodes[nodes.length - 1].getNodeName().replaceAll(".*([0-9]+)$", "$1");
            container.addStringAttribute(attrName, attrValue);
        }
    }
}
