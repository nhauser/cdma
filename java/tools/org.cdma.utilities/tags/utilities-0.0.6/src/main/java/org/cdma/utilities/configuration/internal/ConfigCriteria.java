//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// See AUTHORS file
//******************************************************************************
package org.cdma.utilities.configuration.internal;

import java.util.ArrayList;
import java.util.List;

import org.cdma.interfaces.IDataset;
import org.cdma.utilities.configuration.internal.ConfigParameter.CriterionValue;
import org.jdom2.Element;

/**
 * This class <b>ConfigCriteria</b> gather a set of criterion. It is used to 
 * determine whether a IDataset is matching to <b>ConfigDataset</b>'s criteria.
 * <p>
 * To be considered as matching the IDataset must be returning <b>CriterionValue.TRUE</b>
 * at the evaluation of each <b>ConfigParameter</b>
 * <p>
 * Each <b>ConfigParameter</b> of the contained set of criterion, must return the
 * string form of a <b>CriterionValue</b>. 
 *
 * @see ConfigParameterCriterion ConfigParameterCriterion implements ConfigParameter 
 * @see CriterionValue
 * @author rodriguez
 *
 */
public class ConfigCriteria {
    // Private members
    private List<ConfigParameterCriterion> mCriterion;

    /**
     * Constructor
     * 
     * @note if no ConfigParameterCriterion are added, then it will match any IDataset
     */
    public ConfigCriteria() {
        mCriterion = new ArrayList<ConfigParameterCriterion>();
    }

    /**
     * Parse the DOM element, to add every "if" children element as ConfigParameterCriterion.
     * 
     * @param domCriteria DOM element corresponding to "criteria" section in XML
     */
    public void add(Element domCriteria) {
        // Managing criterion
        if( domCriteria.getName().equals("criteria") ) {
            List<?> nodes = domCriteria.getChildren("if");
            for( Object node : nodes ) {
                mCriterion.add( new ConfigParameterCriterion((Element) node) );
            }
        }
    }

    /**
     * Add the given ConfigParameterCriterion to the set of ConfigParameter.
     * 
     * @param item ConfigParameterCriterion that will be checked when matching 
     */
    public void add(ConfigParameterCriterion item) {
        mCriterion.add(item);
    }

    /**
     * Tells if the given IDataset matches this ConfigCriteria. All clause must be
     * respected by the given IDataset. 
     * 
     * @param dataset to be checked
     * @return boolean true if each ConfigParameterCriterion are respected in the given IDataset
     */
    public boolean match( IDataset dataset ) {
        boolean result = true;
        for( ConfigParameterCriterion criterion : mCriterion ) {
            if( criterion.getValue(dataset).equals(CriterionValue.FALSE.toString()) ) {
                result = false;
                break;
            }
        }
        return result;
    }
    
    @Override
    public String toString() {
        String result = "Criteria: ";
        
        for( ConfigParameterCriterion crit : mCriterion ) {
            result += "\n" + crit;
        }
        
        return result;
    }
}
