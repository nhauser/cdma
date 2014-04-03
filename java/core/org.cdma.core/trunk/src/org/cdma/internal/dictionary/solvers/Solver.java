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
package org.cdma.internal.dictionary.solvers;

/// @cond internal

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.CDMAException;
import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;

/**
 * Solver class is used <b>internally</b> by the <i>Extendend Dictionary mechanism</i>.
 * It permits to define how a key should be resolved to obtain the corresponding CDMA object.
 * According how the solver has been constructed, the result of the solve method will be:
 * - if constructed using a IKey it will produce a LogicalGroup
 * - if constructed using a Path it will produce an IDataItem
 * - if constructed using an IPluginMethod it will return the result of the executed method
 * <p>
 * A Solver can only have one of the following: IPath, Path or IPluginMethod. Each given
 * parameter is exclusive regarding the others.
 * 
 * @author rodriguez
 *
 */

public class Solver {
    private final IKey mKey;             // LogicalGroup to create
    private final Path mPath;            // Physical path to open
    private final IPluginMethod mMethod; // Method call to call
    private final Object[] mParameters;

    public Solver( IKey key ) {
        mKey    = key;
        mPath   = null;
        mMethod = null;
        mParameters = null;
    }

    public Solver( Path path ) {
        mPath   = path;
        mKey    = null;
        mMethod = null;
        mParameters = null;
    }

    public Solver(IPluginMethod method, Object[] parameters) {
        mMethod = method;
        mPath   = null;
        mKey    = null;
        mParameters = parameters;
    }

    public Solver(IPluginMethod method) {
        this(method, null);
    }

    public List<IContainer> solve(Context context) {
        List<IContainer> result = null;

        // Update context with currently executed solver
        context.addSolver(this);

        // If the solver is a path
        if( mPath != null ) {
            // return all found nodes at path
            result = findAllContainerByPath(context);
        }
        // If the solver is a call on a method
        else if( mMethod != null ) {
            context.setParams(mParameters);
            result = executeMethod(context);
        }
        // If the solver is a key create a LogicalGroup
        else if( mKey != null ) {
            result = new ArrayList<IContainer>();
            result.add( new LogicalGroup((LogicalGroup) context.getCaller(), mKey, context.getDataset()) );
        }
        // Return empty list
        else {
            result = new ArrayList<IContainer>();
        }

        return result;
    }


    /**
     * Give an access to the given IKey that constructed this object.
     * @return IKey implementation
     */
    public IKey getKey() {
        return mKey;
    }

    /**
     * Give an access to the given Path that constructed this object.
     * @return Path object
     */
    public Path getPath() {
        return mPath;
    }

    /**
     * Give an access to the given IPluginMethod that constructed this object.
     * @return IPluginMethod implementation
     */
    public IPluginMethod getPluginMethod() {
        return mMethod;
    }

    // ---------------------------------------------------------------
    // PRIVATE methods
    // ---------------------------------------------------------------
    private List<IContainer> findAllContainerByPath(Context context) {
        List<IContainer> result = new ArrayList<IContainer>();

        // Clear the context of previously found nodes
        context.clearContainers();

        // Get the dataset
        IGroup root = context.getDataset().getRootGroup();

        // Try to get all nodes at the targeted path
        try {
            result = root.findAllContainerByPath( mPath.getValue() );
        } catch (NoResultException e) {
            Factory.getLogger().log( Level.WARNING, e.getMessage());
        }

        return result;
    }

    private List<IContainer> executeMethod(Context context) {
        List<IContainer> result = new ArrayList<IContainer>();
        try {
            // Execute the method
            mMethod.execute(context);

            // Get all items added by the method
            result.addAll( context.getContainers() );
        } catch (CDMAException e) {
            Factory.getLogger().log( Level.WARNING, e.getMessage());
        }
        return result;
    }
}

/// @endcond internal
