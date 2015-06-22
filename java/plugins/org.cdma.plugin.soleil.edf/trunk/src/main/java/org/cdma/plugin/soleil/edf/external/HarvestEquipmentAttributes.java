/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.edf.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.plugin.soleil.edf.EdfFactory;
import org.cdma.plugin.soleil.edf.navigation.EdfGroup;
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

    public static String ACQ_SEQUENCE = "acquisition_sequence";
    public static String EQUIPMENT = "equipment";

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
                    setEquipment(container);
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

    private void setEquipment(IContainer container) {
        String equipment = container.getName();
//        if (container instanceof EdfDataItem) {
//            if (container.getParentGroup() != null) {
//                // SOLEIL EDF convention:
//                // A dataitem wich is not directly under the ROOT is contextual data. Its parent is the equipment name.
//                // equipment = container.getParentGroup().getName();
//            }
//        }
        container.addStringAttribute(EQUIPMENT, equipment);
    }

}
