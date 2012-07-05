// ****************************************************************************
// Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors
//    Clement Rodriguez - initial API and implementation
//    Norman Xiong
// ****************************************************************************

/// @cond pluginAPI

/**
 * @brief This IContext interface is used when invoking an external method.
 * 
 * It should contain all required information, so the called method
 * can work properly as if it were in the CDMA.
 * The context is compound of the current dataset, the caller of the 
 * method, the key used to call that method (that can
 * have some parameters), the path (with parameters set) and some
 * parameters that are set by the institute's plug-in.
 */

package org.gumtree.data.dictionary.impl;

import org.gumtree.data.dictionary.IContext;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IKey;

public final class Context implements IContext {
    private IDataset    mDataset;
    private IContainer  mCaller;
    private IKey        mKey;
    private IPath       mPath;
    private Object[]    mParams;

    public Context(IDataset dataset) {
        mDataset = dataset;
        mCaller  = null;
        mKey     = null;
        mPath    = null;
        mParams  = null;
    }

    public Context( IDataset dataset, IContainer caller, IKey key, IPath path ) {
        mDataset = dataset;
        mCaller  = caller;
        mKey     = key;
        mPath    = path;
        mParams  = null;
    }

    @Override
    public String getFactoryName() {
        return mDataset.getFactoryName();
    }

    /**
     * Permits to get the IDataset we want to work on.
     */
    @Override
    public IDataset getDataset() {
        return mDataset;
    }

    /**
     * Permits to set the IDataset we want to work on.
     */
    @Override
    public void setDataset(IDataset dataset) {
        mDataset = dataset;
    }

    /**
     * Permits to get the IContainer that instantiated the context.
     */
    @Override
    public IContainer getCaller() {
        return mCaller;
    }

    /**
     * Permits to set the IContainer that instantiated the context.
     */
    @Override
    public void setCaller(IContainer caller) {
        mCaller = caller;
    }

    /**
     * Permits to get the IKey that lead to this instantiation.
     */
    @Override
    public IKey getKey() {
        return mKey;
    }

    /**
     * Permits to set the IKey that lead to this instantiation.
     */
    @Override
    public void setKey(IKey key) {
        mKey = key;
    }

    /**
     * Permits to get the IPath corresponding to the IKey
     */
    @Override
    public IPath getPath() {
        return mPath;
    }

    /**
     * Permits to set the IPath corresponding to the IKey
     */
    @Override
    public void setPath(IPath path) {
        mPath = path;
    }

    /**
     * Permits to have some parameters that are defined by the instantiating plug-in
     * and that can be useful for the method using this context.
     *  
     * @return array of object
     */
    @Override
    public Object[] getParams() {
        return mParams.clone();
    }

    /**
     * Permits to set some parameters that are defined by the instantiating plug-in
     * and that can be useful for the method using this context.
     *  
     * @return array of object
     */
    @Override
    public void setParams(Object[] params) {
        mParams = params.clone();
    }
}

/// @endcond pluginAPI