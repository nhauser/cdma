//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.external;

import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.internal.dictionary.solvers.AttributeSolver;
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
