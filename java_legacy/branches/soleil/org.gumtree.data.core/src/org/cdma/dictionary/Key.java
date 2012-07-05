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

import java.util.ArrayList;
import java.util.List;

import org.cdma.IFactory;
import org.cdma.dictionary.filter.IFilter;
import org.cdma.interfaces.IKey;

public final class Key implements IKey, Cloneable {
    private String mFactory;
    private String mKey = "";   // key name

    private List<IFilter> mFilters;


    public Key(IFactory factory, String name) {
        mKey     = name;
        mFactory = factory.getName();
        mFilters = new ArrayList<IFilter>();
    }

    public Key(Key key) {
        mKey     = key.getName();
        mFactory = key.getFactoryName();
        mFilters = new ArrayList<IFilter>();
        
        for( IFilter filter : key.mFilters ) {
            mFilters.add(filter);
        }
    }

    @Override
    public String getName() {
        return mKey;
    }

    @Override
    public void setName(String name) {
        mKey = name;
    }

    @Override
    public boolean equals(Object key) {
        if( ! (key instanceof IKey) ) {
            return false;
        }
        else {
            return mKey.equals( ((IKey) key).getName());
        }
    }

    @Override
    public int hashCode() {
        return mKey.hashCode();
    }

    @Override
    public String toString() {
        String result = mKey;
        
        for( IFilter filter : mFilters ) {
            result += "\n  " + filter;
        }
        return result;
    }
    
    @Override
    public Key clone() {
        return new Key(this);
    }

    @Override
    public String getFactoryName() {
        return mFactory;
    }

    @Override
    public int compareTo(Object arg0) {
        return this.getName().compareTo(arg0.toString());
    }
    
    @Override
    public List<IFilter> getFilterList() {
        return mFilters;
    }

    @Override
    public void pushFilter(IFilter filter) {
        mFilters.add(filter);
    }

    @Override
    public IFilter popFilter() {
        if( mFilters.size() > 0) {
            return mFilters.remove(0);
        }
        else {
            return null;
        }
    }
}
