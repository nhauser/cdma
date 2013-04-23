// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Clément Rodriguez - initial API and implementation
// ****************************************************************************
package org.cdma.internal.dictionary.solvers;

/// @cond internal

import java.util.List;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.Context;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;


/**
 * @brief AttributeSolver aims to return an IAttribute that corresponds to the mapping.
 * 
 * The AttributeSolver class as an unique method which solve. The goal is to return
 * the IAttribute that is defined into the institute's mapping. To do so, it uses
 * a Context which permits to access the CDMA environment. The IAttribute can be 
 * searched in a specific path, or might need a named method execution a to construct it.
 * <p>
 * That class is used internally by the ItemSolver class to construct attributes.
 * 
 * @see org.cdma.dictionary.IPluginMethod
 * @see org.cdma.dictionary.Context
 * 
 * @author rodriguez
 *
 */

public class AttributeSolver {
    List<Solver> mSolver;   // List of solvers to process to get IContainer attributes
    String       mName;     // Attribute name
    String       mValue;
    
    public AttributeSolver( String name, List<Solver> solvers ) {
        mName    = name;
        mSolver  = solvers;
        mValue   = null;
    }
    
    public AttributeSolver(String name, String value) {
    	mName   = name;
    	mValue  = value;
    	mSolver = null;
	}

	/**
     * Return a IAttribute generated using the given Context.
     * 
     * @param context of attribute resolution
     * @return IAttribute 
     */
    public IAttribute solve( Context context ) {
        IAttribute attribute = null;
        List<IContainer> list = null;
        
        // Give this attribute solver as a parameter of the context
        //context.setParams( new AttributeSolver[] {this} );
        
        if( mValue != null ) {
        	String plugin = context.getCaller().getFactoryName();
        	IFactory factory = Factory.getFactory(plugin);
        	if( factory != null ) {
        		attribute = factory.createAttribute(mName, mValue);
        	}
        }
        else {
	        // Get IContainer matching to this solver to seek the named attribute
	        for( Solver solver : mSolver ) {
	            list = solver.solve(context);
	            context.setContainers(list);
	        }
	        
	        // Return the named IAttribute of the found IContainer
	        if( list != null && ! list.isEmpty() ) {
	            attribute = list.get(0).getAttribute(mName);
	        }
        }
        return attribute;
    }
    
    public List<Solver> getSolvers() {
        return mSolver;
    }
    
    public String getName() {
        return mName;
    }
}

/// @endcond internal