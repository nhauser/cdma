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
package org.cdma.plugin.soleil.edf.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IContainer;
import org.cdma.math.IArrayMath;
import org.cdma.plugin.soleil.edf.EdfFactory;
import org.cdma.plugin.soleil.edf.navigation.EdfDataItem;
import org.cdma.utils.Utilities.ModelType;

public class ModifyData implements IPluginMethod {


    private enum Operations {
        ADD, SUBSTRACT, MULTIPLY, DIVIDE;
    }

    @Override
    public String getFactoryName() {
        return EdfFactory.NAME;
    }

    @Override
    public void execute(final Context context) throws CDMAException {
        List<IContainer> inList = context.getContainers();
        List<IContainer> outList = new ArrayList<IContainer>();

        EdfDataItem item;

        String operationAsString = (String) context.getParams()[0];
        Operations operation = Operations.valueOf(operationAsString);

        String correctionFactorAsString = (String) context.getParams()[1];
        double correctionFactor = Double.valueOf(correctionFactorAsString);

        if (operation != null && correctionFactorAsString != null) {

            for (IContainer container : inList) {
                if (container.getModelType().equals(ModelType.DataItem)) {
                    item = (EdfDataItem) container;

                    IArray compositeArray = item.getData();

                    IArrayMath maths = compositeArray.getArrayMath();
                    IArrayMath resultMath = null;

                    if (operation.equals(Operations.ADD)) {
                        resultMath = maths.toAdd(correctionFactor);
                    } else if (operation.equals(Operations.SUBSTRACT)) {
                        resultMath = maths.toAdd(-correctionFactor);
                    } else if (operation.equals(Operations.MULTIPLY)) {
                        resultMath = maths.toScale(correctionFactor);
                    } else if (operation.equals(Operations.DIVIDE)) {
                        resultMath = maths.toScale(1 / correctionFactor);
                    }

                    item.setCachedData(resultMath.getArray(), false);
                    outList.add(item);
                }
            }
        }
        // Update context
        context.setContainers(outList);
    }
}
