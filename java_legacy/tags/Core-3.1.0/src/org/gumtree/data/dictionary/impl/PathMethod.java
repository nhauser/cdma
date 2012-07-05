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
package org.gumtree.data.dictionary.impl;

/// @cond clientAPI

/**
 * @brief The IPathMethod interface refers to a method that should be executed to obtain the requested data.
 * 
 * This interface is only used by the extended dictionary mechanism. Its aim
 * is to allow mapping dictionary to specify how to get a IDataItem that don't
 * only rely on a specific path. The IPathMethod permits to specify a method that
 * must be executed associated to a IPath. The more often it's called because we have
 * to pre/post process data to fit a particular need. 
 * <p>
 * The method will be called using an already implemented mechanism. The call will
 * be done only when the IPath, carrying it, will be resolved by the ILogicalGroup.
 * <p>
 * @example In case of a stack of spectrums that are split onto various IDataItem
 * if we want to see them as only one IDataItem the only solution will be to use a
 * specific method for the request.
 * 
 * @author rodriguez
 */

import java.util.ArrayList;

import org.gumtree.data.dictionary.IPathMethod;

public class PathMethod implements IPathMethod  {
    String            m_method;
    ArrayList<Object> m_param;
    boolean           m_external = false; // Is the method from the core or is it an external one 

    public PathMethod(String method) { m_method = method; m_param = new ArrayList<Object>(); }
    public PathMethod(String method, Object param) { m_method = method; m_param = new ArrayList<Object>(); m_param.add(param); }

    /**
     * Returns name of the method that will be called (using it's package name)
     * return String
     */
    @Override
    public String getName() { return m_method; }

    /**
     * Sets the name of the method that will be called (using it's package name)
     * 
     * @param method in its namespace
     */
    @Override
    public void setName(String method) { m_method = method; }

    /**
     * Return parameters Object that are used by this method
     * 
     * @return Object array
     */
    @Override
    public Object[] getParam() { return m_param.toArray(); }

    /**
     * Set a parameter value that will be used by this method
     * 
     * @param param Object that will be used by this method
     * @note works as a FIFO
     */
    @Override
    public void pushParam(Object param) { m_param.add(param); }

    /**
     * Set a parameter value that will be used by this method
     * 
     * @return Object that will be used by this method
     * @note works as a FIFO
     */
    @Override
    public Object popParam() { return m_param.remove(m_param.size()); }

    /**
     * Tells whether or not the method is already contained by the plug-in or if it 
     * will be dynamically loaded from the external folder specific to the plug-in.
     * 
     * @return boolean
     */
    @Override
    public boolean isExternalCall() { return m_external; }

    /**
     * Set whether or not the method is already contained by the plug-in or if it 
     * will be dynamically loaded from the external folder specific to the plug-in.
     * 
     * @return boolean
     * @see LogicalGroup.resolveMethod
     * @see org.gumtree.data.dictionary.IClassLoader
     */
    @Override
    public void isExternal(boolean external) { m_external = external; }
}

/// @endcond clientAPI