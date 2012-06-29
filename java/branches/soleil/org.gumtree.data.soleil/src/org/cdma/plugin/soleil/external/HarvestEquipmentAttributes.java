package org.cdma.plugin.soleil.external;

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
 * For each container in the context, this method will try to find its meta-data.
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
        List<IContainer> list = context.getContainers();
        
        for( IContainer container : list ) {
            ModelType type = container.getModelType();
            
            switch( type ) {
                case Group: {
                    NxsGroup group = (NxsGroup) container;
                    IGroup root = container.getDataset().getRootGroup();
                    NexusNode[] nodes = group.getPathNexus().getNodes();
                    
                    setAttributeAcquisitionSequence(container, nodes, root);
                    setAttributeEquipment(container, nodes, root);
                    
                    break;
                }
                case DataItem: {
                    NxsDataItem item = (NxsDataItem) container;
                    IGroup root = container.getDataset().getRootGroup();
                    
                    // Try to set attributes
                    DataItem[] n4tItems = item.getNexusItems();
                    NexusNode[] nodes = n4tItems[0].getPath().getNodes();
                    
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
    }

    
    void setAttributeAcquisitionSequence( IContainer container, NexusNode[] nodes, IGroup root ) {
        // Scan attribute 
        if( nodes.length > 1 &&  nodes[0].getClassName().equals("NXentry")) {
            String attrName = "acquisition_sequence";
            String attrValue = root.getGroup( nodes[0].getNodeName() ).getShortName();
            container.addStringAttribute( attrName, attrValue );
        }
    }
    
    void setAttributeEquipment( IContainer container, NexusNode[] nodes, IGroup root ) {
        // Equipment attribute
        if( nodes.length > 2 && nodes[1].getClassName().equals("NXinstrument") ) {
            String attrName = "equipment";
            String attrValue = root.getGroup( nodes[0].getNodeName() ).getGroup( nodes[1].getNodeName() ).getGroup( nodes[2].getNodeName() ).getShortName();
            container.addStringAttribute( attrName, attrValue );
        }
    }
}
