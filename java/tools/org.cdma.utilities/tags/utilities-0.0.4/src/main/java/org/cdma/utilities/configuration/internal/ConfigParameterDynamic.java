/*******************************************************************************
 * Copyright (c) 2012 Synchrotron SOLEIL.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
 ******************************************************************************/
package org.cdma.utilities.configuration.internal;

import java.io.IOException;

import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.utilities.configuration.internal.ConfigParameter.CriterionType;
import org.jdom2.Attribute;
import org.jdom2.Element;

/**
 * <b>ConfigParameterDynamic implements ConfigParameter</b><br>
 * 
 * The aim of that class is to define <b>parameters that are dependent</b> of the content 
 * of the IDataset. It means that those parameters <b>are specific to that file</b>.
 * The evaluation will be performed on the fly when requested.<p> 
 * 
 * This class is used by ConfigDataset when the plug-in asks for a specific value
 * for that specific file. For example is:<br/>
 * - how to know on which beamline it was created</br>
 * - what the data model is</br><p>
 * 
 * <b>The parameter's type can be:</b><br/>
 * - CriterionType.EXIST will try to find a specific path in the IDataset <br/>
 * - CriterionType.NAME get the name of the object targeted by the path<br/>
 * - CriterionType.VALUE get the value of IDataItem targeted by the path<br/>
 * - CriterionType.CONSTANT the value will be a constant<br/>
 * - CriterionType.EQUAL will compare values targeted by a path to a referent one<p>
 * 
 * It corresponds to "parameter" DOM element in the "parameters" section
 * of the plug-in's configuration file.
 *
 * @see ConfigParameter
 * @see CriterionType
 * 
 * @author rodriguez
 */
public final class ConfigParameterDynamic implements ConfigParameter {
    private String         mName;        // Parameter name
    private String         mPath;        // Path to seek in
    private CriterionType  mType;        // Type of operation to do
    private CriterionValue mTest;        // Expected property
    private String         mValue;       // Comparison value with the expected property


    public ConfigParameterDynamic(Element parameter) {
        init(parameter);
    }

    @Override
    public CriterionType getType() {
        return mType;
    }

    public String getName() {
        return mName;
    }

    public String getValue(IDataset dataset) {
        String result = "";
        IContainer cnt;
        switch( mType ) {
        case EXIST:
            cnt = openPath(dataset);
            CriterionValue crt;
            if( cnt != null ) {
                crt = CriterionValue.TRUE;
            }
            else {
                crt = CriterionValue.FALSE;
            }
            result = mTest.equals( crt ) ? CriterionValue.TRUE.toString() : CriterionValue.FALSE.toString();
            break;
        case NAME:
            cnt = openPath(dataset);
            if( cnt != null ) {
                result = cnt.getShortName();
            }
            else {
                result = null;
            }
            break;
        case VALUE:
            cnt = openPath(dataset);
            if( cnt != null && cnt instanceof IDataItem ) {
                try {
                    result = ((IDataItem) cnt).getData().getObject(((IDataItem) cnt).getData().getIndex()).toString();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            else {
                result = null;
            }
            break;
        case CONSTANT:
            result = mValue;
            break;
        case EQUAL:
            cnt = openPath(dataset);
            if( cnt != null && cnt instanceof IDataItem ) {
                String value;
                try {
                    value = ((IDataItem) cnt).getData().getObject(((IDataItem) cnt).getData().getIndex()).toString();
                    if( value.equals(mValue) ) {
                        crt = CriterionValue.TRUE;
                    }
                    else {
                        crt = CriterionValue.FALSE;
                    }
                    result = mTest.equals( crt ) ? CriterionValue.TRUE.toString() : CriterionValue.FALSE.toString();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            else {
                result = null;
            }
            break;
        case NONE:
        default:
            result = null;
            break;
        }
        return result;
    }

    @Override
    public String toString() {
        String result = "Name: " + mName +
                        " Type: " + mType + 
                        " Path: " + mPath + 
                        " Test: " + mTest +
                        " Value: " + mValue;
        return result;
    }
    
    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    private void init(Element dom) {
        // Given element is <parameter>
        if( dom.getName().equals("parameter") )
        {
            mName = dom.getAttributeValue("name");
            mPath = dom.getAttributeValue("target");
            
            Attribute attribute;
            String test;
    
            attribute = dom.getAttribute("type");
            if( attribute != null )
            {
                mType  = CriterionType.valueOf(attribute.getValue().toUpperCase());
                String value = dom.getAttributeValue("constant");
                if( value != null ) {
                    mValue = value;
                }
    
                test = dom.getAttributeValue("test");
                if( test != null ) {
                    mTest = CriterionValue.valueOf(test.toUpperCase());
                }
                else {
                    mTest = CriterionValue.NONE;
                }
            }
        }
    }

    private IContainer openPath( IDataset dataset ) {
        IGroup root = dataset.getRootGroup();
        IContainer result = null;
        try {
            result = root.findContainerByPath(mPath);
        } catch (NoResultException e) {
        }
        return result;
    }
}
