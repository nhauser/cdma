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
package org.cdma.dictionary;

/// @cond clientAPI

/**
 * @brief The Path class defines a destination that will be interpreted by the plug-in.
 * 
 * A Path describes how to reach a specific item in the IDataset. The way that path
 * are written is format plug-in dependent. The more often it will be a <code>String</code>
 * with a separator between nodes that are relevant for the plug-in.
 * <p>
 * The String representation will only be interpreted by the plug-in.
 * 
 * @author rodriguez
 * @see org.cdma.interfaces.IKey
 */

import org.cdma.IFactory;
import org.cdma.internal.IModelObject;

public class Path implements IModelObject, Cloneable {
    String           m_factory;
    String           m_pathValue;

    public Path(IFactory factory) {
        m_factory      = factory.getName();
        m_pathValue    = null;
    }

    public Path(IFactory factory, String path) {
        m_factory      = factory.getName();
        m_pathValue    = path;
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
    public String getValue() {
        return m_pathValue;
    }

    /**
     * Set the value of the path in string
     * 
     * @param path string representation of the targeted node in a IDataset
     */
    public void setValue(String path) {
        m_pathValue  = path;
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
}

/// @endcond clientAPI