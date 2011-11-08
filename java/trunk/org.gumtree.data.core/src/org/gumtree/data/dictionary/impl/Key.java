/****************************************************************************** 
 * Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 * 	  Clement Rodriguez - initial API and implementation
 *    Norman Xiong
 ******************************************************************************/
package org.gumtree.data.dictionary.impl;

import java.util.ArrayList;
import java.util.List;

import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.interfaces.IKey;

public class Key implements IKey {
	private String m_factory;
    private String m_key = "";   // key name
    private List<IPathParameter> m_filters;
    
    public Key(IFactory factory, String name) {
    	m_key     = name;
    	m_factory = factory.getName();
        m_filters = new ArrayList<IPathParameter>();
    }
    
    public Key(IKey key) {
    	m_key     = key.getName();
    	m_factory = key.getFactoryName();
        m_filters = new ArrayList<IPathParameter>();
        for( IPathParameter param : key.getParameterList() ) {
        	m_filters.add(param.clone());
        }
    }
    
    @Override
    final public List<IPathParameter> getParameterList() {
        return m_filters;
    }
    
    @Override
    public String getName() {
        return m_key;
    }

    @Override
    public void setName(String name) {
        m_key = name;
    }

    @Override
    public boolean equals(IKey key) {
    	return m_key.equals(key.getName());
    }
    
    @Override
    public boolean equals(Object key) {
    	if( ! (key instanceof IKey) ) {
    		return false;
    	}
    	else {
    		return this.equals( (IKey) key);
    	}
    }
    
    @Override
    public int hashCode() {
    	return m_key.hashCode();
    }
    
	@Override
    public String toString() {
        return m_key;
    }

    @Override
    public void pushParameter(IPathParameter filter) {
        m_filters.add(filter);
    }

    @Override
    public IPathParameter popParameter() {
        if( m_filters.size() > 0) {
            return m_filters.remove(0);
        }
        else {
            return null;
        }
    }
    
    @Override
    public IKey clone() {
    	IKey key = Factory.getFactory(m_factory).createKey(m_key);
    	for( IPathParameter filter : m_filters ) {
    		key.pushParameter( filter.clone() );
    	}
    	return key;
    }
    
	@Override
	public String getFactoryName() {
		return m_factory;
	}

	@Override
	public int compareTo(Object arg0) {
		return this.getName().compareTo(arg0.toString());
	}
}
