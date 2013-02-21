// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    Tony Lam (nxi@Bragg Institute) - initial API and implementation
//    Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
// ****************************************************************************
package org.cdma.dictionary;

import java.util.ArrayList;
import java.util.List;

import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IKey;
import org.cdma.internal.dictionary.solvers.Solver;

public final class Context {
    private IDataset         mDataset;
    private IContainer       mCaller;
    private IKey             mKey;
    private Object[]         mParams;
    private List<IContainer> mContainers;
    private List<Solver>     mSolvers;
    private Concept          mConcept;

    public Context(IDataset dataset) {
        mDataset    = dataset;
        mCaller     = null;
        mKey        = null;
        mSolvers    = new ArrayList<Solver>();
        mParams     = null;
        mContainers = new ArrayList<IContainer>();
    }

    public Context( IDataset dataset, IContainer caller, IKey key ) {
        mDataset    = dataset;
        mCaller     = caller;
        mKey        = key;
        mSolvers    = new ArrayList<Solver>();
        mParams     = null;
        mContainers = new ArrayList<IContainer>();
        mConcept    = null;
    }

    /**
     * Permits to get the IDataset we want to work on.
     */
    public IDataset getDataset() {
        return mDataset;
    }

    /**
     * Permits to set the IDataset we want to work on.
     */
    public void setDataset(IDataset dataset) {
        mDataset = dataset;
    }

    /**
     * Permits to get the IContainer that instantiated the context.
     */
    public IContainer getCaller() {
        return mCaller;
    }

    /**
     * Permits to set the IContainer that instantiated the context.
     */
    public void setCaller(IContainer caller) {
        mCaller = caller;
    }

    /**
     * Permits to get the IKey that lead to this instantiation.
     */
    public IKey getKey() {
        return mKey;
    }

    /**
     * Permits to set the IKey that lead to this instantiation.
     */
    public void setKey(IKey key) {
        mKey = key;
    }

    /**
     * Permits to get the Solver list corresponding to the 
     * IKey that have been previously executed
     */
    public List<Solver> getSolver() {
        return mSolvers;
    }

    /**
     * Permits to add a Solver corresponding to the IKey
     */
    public void addSolver(Solver solver) {
        mSolvers.add( solver );
    }

    /**
     * Permits to have some parameters that are defined by the instantiating plug-in
     * and that can be useful for the method using this context.
     * 
     * @return array of object
     */
    public Object[] getParams() {
        return mParams.clone();
    }

    /**
     * Permits to set some parameters that are defined by the instantiating plug-in
     * and that can be useful for the method using this context.
     *  
     * @return array of object
     */
    public void setParams(Object[] params) {
        mParams = params.clone();
    }
    
    /**
     * Get the list of found Containers by previously executed solver
     * 
     * @return list of containers
     */
    public List<IContainer> getContainers() {
        return mContainers;
    }

    /**
     * Clear the list of found Containers
     */
    public void clearContainers() {
        mContainers.clear();
    }
    
    /**
     * Set the list of found containers
     */
    public void setContainers( List<IContainer> items ) {
        mContainers = items;
    }
    
    /**
     * Add a Container to the list of found Containers
     */
    public void addContainer( IContainer item ) {
        mContainers.add(item);
    }
    
    /**
     * Returns the concept that is expected for this context.
     * 
     * @return concept that the system expects to have
     */
    public Concept getConcept() {
        return mConcept;
    }
    
    /**
     * Set the concept that is expected by the context user.
     * 
     * @param concept to have at end of the process
     */
    public void setConcept(Concept concept) {
        mConcept = concept;
    }
}

/// @endcond pluginAPI