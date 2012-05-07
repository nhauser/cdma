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
 * @brief The IPath interface defines a destination that will be interpreted by the plug-in.
 * 
 * A IPath is describes how to reach a specific item in the IDataset. The way that path
 * are written is format plug-in dependent. The more often it will be a <code>String</code>
 * with a separator between nodes that are relevant for the plug-in.
 * <p>
 * The String representation will only be interpreted by the plug-in. But it can be modified
 * using some parameters to make selective choice while browsing the nodes structure of the
 * IDataset.
 * <p>
 * In other cases (for the extended dictionary mechanism) it can also be a call on a plug-in 
 * specific method. It permits to conform to standardized way of returning a data item. For instance
 * it can be returning a stack of spectrums that are split among several nodes. 
 * 
 * @author rodriguez
 * @see org.gumtree.data.interfaces.IKey
 * @see org.gumtree.data.dictionary.IPathParameter
 * @see org.gumtree.data.dictionary.IPathMethod
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.IPathMethod;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.utils.Utilities.ParameterType;

public class Path implements IPath {
    final public String PARAM_PATTERN = "\\$\\(([^\\)]*)\\)"; // parameters have following shape "/my/path/$(parameter)/my_node"
    final private String PATH_SEPARATOR;

    String            m_factory;
    String            m_pathValue;
    String            m_pathOrigin;
    List<IPathMethod> m_methods;

    public Path(IFactory factory) {
        m_factory      = factory.getName();
        m_pathValue    = null;
        m_pathOrigin   = null;
        m_methods      = new ArrayList<IPathMethod>();
        PATH_SEPARATOR = factory.getPathSeparator();
    }

    public Path(IFactory factory, String path) {
        m_factory      = factory.getName();
        m_pathOrigin   = path;
        m_pathValue    = path;
        m_methods      = new ArrayList<IPathMethod>();
        PATH_SEPARATOR = factory.getPathSeparator();
    }

    /**
     * Returns the String representation of the path.
     */
    @Override
    public String toString() {
        return m_pathValue;
    }

    @Override
    public String getFactoryName() {
        return null;
    }

    /**
     * Get the value of the path
     * 
     * @return string path value
     */
    @Override
    public String getValue() {
        return m_pathValue;
    }

    /**
     * Getter on methods that should be invoked to get data.
     * The returned list is unmodifiable.
     *  
     * @return unmodifiable list of methods
     */
    @Override
    public List<IPathMethod> getMethods() {
        return Collections.unmodifiableList(new ArrayList<IPathMethod>(m_methods));
    }

    /**
     * Set methods that should be invoked to get data.
     * The list contains PathMethod having method and array of Object for
     * arguments.
     * 
     * @param list that will be copied to keep it unmodifiable
     * @return unmodifiable map of methods
     */
    @Override
    public void setMethods(List<IPathMethod> methods) {
        m_methods = methods;
    }

    /**
     * Set the value of the path in string
     * 
     * @param path string representation of the targeted node in a IDataset
     */
    @Override
    public void setValue(String path) {
        m_pathOrigin = path;
        m_pathValue  = path;
    }

    /**
     * Will modify the path to make given parameters efficient.
     * 
     * @param parameters list of parameters to be inserted
     */
    @Override
    public void applyParameters(List<IPathParameter> params) {
        for( IPathParameter param : params ) {
            m_pathValue = m_pathValue.replace( "$("+ param.getName() + ")" , param.getValue().toString() );
        }
    }

    /**
     * Will modify the path to remove all traces of parameters 
     * that are not defined.
     */
    @Override
    public void removeUnsetParameters() {
        m_pathValue = m_pathValue.replaceAll( PARAM_PATTERN , "");
    }

    /**
     * Will modify the path to unset all parameters that were defined
     */
    @Override
    public void resetParameters() {
        m_pathValue = m_pathOrigin;
    }

    /**
     * Clone the path so it keep unmodified when updated while it
     * has an occurrence in a dictionary
     * 
     * @return a clone this path
     */
    @Override
    public Object clone() {
        try {
            return super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Analyze the path to reach the first undefined parameter.
     * Return the path parameter to open the node. The parameter has 
     * a wildcard for value (i.e: all matching nodes can be opened) 
     * 
     * @param param output path that will be updated with the appropriate node's
     *           type and name until to reach the first path parameter
     * 
     * @return IPathParameter 
     *        having the right type and name and a wildcard for value (empty)
     */
    @Override
    public IPathParameter getFirstPathParameter(StringBuffer output) {
        IPathParameter result = null;
        String[] pathParts = m_pathValue.split(Pattern.quote(PATH_SEPARATOR));
        String name;

        // Split the path into nodes
        for( String part : pathParts ) {
            if( part != null && !part.isEmpty() ) {
                output.append(PATH_SEPARATOR);
                Pattern pattern = Pattern.compile(PARAM_PATTERN);
                Matcher matcher = pattern.matcher(part);
                if( matcher.find() ) {
                    name = part.replaceAll(".*" + PARAM_PATTERN + ".*", "$1");
                    result = Factory.getFactory(m_factory).createPathParameter(ParameterType.SUBSTITUTION, name, "");
                    output.append(part.replaceAll( PARAM_PATTERN , "") );
                    break;
                }
                else {
                    output.append(part);
                }
            }
        }

        return result;
    }

}

/// @endcond clientAPI