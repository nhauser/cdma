//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.nexus.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.plugin.soleil.nexus.NxsFactory;
import org.cdma.plugin.soleil.nexus.utils.NxsConstant;

public class SelectNodeByRegionAttribute implements IPluginMethod {

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public void execute(Context context) throws CDMAException {
        IContainer nodeToApplyAttribute = (IContainer) context.getParams()[0];
        String region = nodeToApplyAttribute.getAttribute(NxsConstant.ATTR_REGION).getStringValue();
        
        List<IContainer> instruments = context.getContainers();
        List<IContainer> result = new ArrayList<IContainer>();
        
        for( IContainer instrument : instruments ) {
            IAttribute attr = instrument.getAttribute(NxsConstant.ATTR_REGION);
            if( attr != null && region.equals( attr.getStringValue() ) ) {
                result.add(instrument);
                break;
            }
        }
        
        context.setContainers(result);
    }

}
