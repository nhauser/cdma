//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.edf.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.plugin.edf.EdfFactory;
import org.cdma.plugin.edf.navigation.EdfGroup;
import org.cdma.utils.Utilities.ModelType;

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

    private static String ACQ_SEQUENCE = "acquisition_sequence";

    @Override
    public String getFactoryName() {
        return EdfFactory.NAME;
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
                    setAttributeAcquisitionSequence(container);
                    break;
                }
                case DataItem: {
                    setAttributeAcquisitionSequence(container);
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

    private void setAttributeAcquisitionSequence(IContainer container) {
        EdfGroup root = (EdfGroup) container.getRootGroup();
        String attrValue = root.getShortName();
        container.addStringAttribute(ACQ_SEQUENCE, attrValue);
    }
}
