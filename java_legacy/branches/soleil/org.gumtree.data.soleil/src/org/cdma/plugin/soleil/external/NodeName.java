package org.cdma.plugin.soleil.external;

import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.internal.dictionary.AttributeSolver;
import org.cdma.plugin.soleil.NxsFactory;

/**
 * Generate an IAttribute having:
 * - the name given by the AttributeSolver in Context's parameters
 * - the value is found in the IContainer list given thru the Context
 * 
 * @author rodriguez
 *
 */

public class NodeName implements IPluginMethod {

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> found = context.getContainers();
        String name = ( (AttributeSolver[]) context.getParams() )[0].getName();
        
        for( IContainer container : found ) {
            container.addStringAttribute(name, container.getShortName() );
        }
    }

}
