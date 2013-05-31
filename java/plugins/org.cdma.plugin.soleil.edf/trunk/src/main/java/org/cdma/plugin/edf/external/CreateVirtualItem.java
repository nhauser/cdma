package org.cdma.plugin.edf.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.arrays.DefaultArrayMatrix;
import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.edf.EdfFactory;
import org.cdma.plugin.edf.navigation.EdfDataItem;
import org.cdma.utils.Utilities.ModelType;

public class CreateVirtualItem implements IPluginMethod {

    @Override
    public String getFactoryName() {
        return EdfFactory.NAME;
    }

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> inList = context.getContainers();
        List<IContainer> outList = new ArrayList<IContainer>();

        EdfDataItem item;

        DefaultArrayMatrix array;
        String name;
        for (IContainer container : inList) {
            if (container.getModelType().equals(ModelType.Group)) {
                name = container.getName();

                array = new DefaultArrayMatrix(EdfFactory.NAME, name.toCharArray());
                item = new EdfDataItem(EdfFactory.NAME, array);
                item.setName(name);
                item.setShortName(container.getShortName());

                item.setParent((IGroup) container);
                item.setCachedData(array, false);
                for (IAttribute attr : container.getAttributeList()) {
                    item.addOneAttribute(attr);
                }
                outList.add(item);
            }
            else {
                outList.add(container);
            }
        }

        // Update context
        context.setContainers(outList);
    }

}