package org.cdma.plugin.soleil.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.navigation.NxsDataItem;
import org.cdma.utils.Utilities.ModelType;

/**
 * Create a list of IDataItem that are empty from a IGroup list. Created items have no data linked.
 * 
 * @param context
 * @throws CDMAException
 */

public class CreateVirtualItem implements IPluginMethod {

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> inList = context.getContainers();
        List<IContainer> outList = new ArrayList<IContainer>();
        
        NxsDataItem item;
        for( IContainer container : inList ) {
            if( container.getModelType().equals( ModelType.Group ) ) {
                item = new NxsDataItem();
                item.setName( container.getName() );
                item.setShortName( container.getShortName() );
                item.setDataset( container.getDataset() );
                for( IAttribute attr : container.getAttributeList() ) {
                    item.addOneAttribute(attr);
                }
                outList.add(item);
            }
            else {
                outList.add( container );
            }
        }
        
        // Update context
        context.setContainers(outList);
    }

}
