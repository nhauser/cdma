/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.utils.NxsConstant;

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

        for (IContainer instrument : instruments) {
            IAttribute attr = instrument.getAttribute(NxsConstant.ATTR_REGION);
            if (attr != null && region.equals(attr.getStringValue())) {
                result.add(instrument);
                break;
            }
        }

        context.setContainers(result);
    }

}
