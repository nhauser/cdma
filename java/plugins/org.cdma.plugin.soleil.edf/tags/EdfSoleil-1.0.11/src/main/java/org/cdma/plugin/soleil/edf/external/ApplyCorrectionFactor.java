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

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.plugin.soleil.edf.EdfFactory;
import org.cdma.plugin.soleil.edf.array.InlineArray;
import org.cdma.plugin.soleil.edf.navigation.EdfDataItem;
import org.cdma.utils.Utilities.ModelType;

public class ApplyCorrectionFactor implements IPluginMethod {

    @Override
    public String getFactoryName() {
        return EdfFactory.NAME;
    }

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> inList = context.getContainers();
        List<IContainer> outList = new ArrayList<IContainer>();

        EdfDataItem item;

        String correctionFactorAsString = (String) context.getParams()[0];
        double correctionFactor = Double.valueOf(correctionFactorAsString);

        for (IContainer container : inList) {
            if (container.getModelType().equals(ModelType.DataItem)){
                item = (EdfDataItem) container;
                Object inlineOriginalArray = item.getData().getArrayUtils().copyTo1DJavaArray();
                int length = Array.getLength(inlineOriginalArray);
                Object finalArray = Array.newInstance(double.class, length);

                for(int i = 0 ; i < length ; i++){
                    String strValue = String.valueOf(Array.get(inlineOriginalArray, i));
                    double singleValue = Double.valueOf(strValue);
                    Array.set(finalArray, i, singleValue * correctionFactor);
                }
                InlineArray inlineArray = new InlineArray("EDF", finalArray, new int[] { length, 1 });
                item.setCachedData(inlineArray, false);
                outList.add(item);
            }
        }

        // Update context
        context.setContainers(outList);
    }
}
