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

/// @cond internal

/**
 * @brief The IPathParameter interface is used to make a selective choice when browsing a IDataset.
 * 
 * A IPathParameter represents conditions that permits identifying a specific node using the 
 * extended dictionary mechanism. 
 * When according to a given IPath several IContainer can be returned, the path parameter
 * will make possible how to find which one is relevant.
 * <p>
 * The parameter can consist in a regular expression on a name, an attribute or 
 * whatever that should be relevant to formally identify a specific node while 
 * several are possible according to the path.
 * 
 * @see org.gumtree.data.utils.Utilities.ParameterType
 * @see org.gumtree.data.dictionary.IPath
 * 
 * @author rodriguez
 *
 */

import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.utils.Utilities.ParameterType;

public class PathParameter implements IPathParameter {
    private ParameterType  m_type;
    private Object         m_value;
    private String         m_name;
    private String         m_factory;

    public PathParameter(IFactory factory, ParameterType filter, String name) {
        m_name    = name;
        m_type    = filter;
        m_value   = null;
        m_factory = factory.getName();
    }

    public PathParameter(IFactory factory, ParameterType filter, String name, Object value) {
        m_type    = filter;
        m_value   = value;
        m_name    = name;
        m_factory = factory.getName();
    } 

    /**
     * Get the filter's kind
     * 
     * @return filter's kind
     */
    @Override
    public ParameterType getType() {
        return m_type;
    }

    /**
     * Get the filter's value
     * 
     * @return filter's value
     */
    @Override
    public Object getValue() {
        return m_value;
    }

    /**
     * Set the filter's value
     * 
     * @param value of the filter
     */
    @Override
    public void setValue(Object value) {
        m_value = value;
    }

    /**
     * Equality test
     * 
     * @return true if both KeyFilter have same kind and value
     */
    @Override
    public boolean equals(IPathParameter keyfilter) {
        return ( m_value.equals(keyfilter.getValue()) && m_type.equals(keyfilter.getType()) ) ;
    }

    /**
     * To String method
     * 
     * @return a string representation of the KeyFilter
     */
    @Override
    public String toString() {
        return m_name + "=" + m_value; 
    }

    /**
     * Clone this IKeyFilter
     * @return a copy of this
     */
    @Override
    public IPathParameter clone() {
        PathParameter param = new PathParameter();
        param.m_factory = m_factory;
        param.m_name    = m_name;
        param.m_type    = m_type;
        param.m_value   = m_value;
        return param;
    }

    /**
     * Get the filter's name
     * 
     * @return name of the filter
     */
    @Override
    public String getName() {
        return m_name;
    }

    @Override
    public String getFactoryName() {
        return m_factory;
    }

    /*
  @Override
  public void update(IPath path) {
    String value = path.getValue();
    value = value.replaceAll("\\$\\(" + m_name + "\\)", m_value.toString() );
    path.setValue(value);
  }
     */
    private PathParameter() {};
}


/// @endcond internal