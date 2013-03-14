//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.utilities.performance;

import java.text.Collator;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeSet;

/**
 * @brief The Buffer class stores in memory the given data associated to a String.
 * 
 * The aim is to prevent recalculating several time things that have been done yet.
 * <p>
 * Each object are associated to 'path'. When the buffer is starting to get
 * full, its lighter half is automatically flushed. Indeed, a cost in time is generated,
 * while accessing each path. When a flush is needed the costlier path are kept,
 * whereas the other are removed.
 * 
 * @author rodriguez
 */

public class Buffer<U, V> {

	private Comparator<V> mCollator;
	
    // Maximum size of the buffer (number of available slots)
    private int mBufferSize;

    // HashMap containing all object for a specific key (key => [collection of values])
    private final Map<U, Collection<V>> mKeyValues;

    // HashMap containing key usage count (used to remove less used keys when cleaning buffer)
    private final Map<U, Integer> mKeyUsageWeigth;

    // Constructor
    /**
     * Construct a Buffer with given maximum size and collator to sort the buffered objects.
     * @param size of the buffer 
     * @param collator to sort buffered set
     * @note if 'size' is negative then auto free mechanism is disabled
     * @note if 'collator' is null then sorting is disabled
     */
    public Buffer(int size, Comparator<V> comparator) {
    	mKeyUsageWeigth = new HashMap<U, Integer>();
    	mKeyValues      = new HashMap<U, Collection<V> >();
        mBufferSize     = size;
        mCollator       = comparator; 
    }
    
    public Buffer(int size) {
        this(size, null);
    }

    /**
     * Returns the current maximum size of the buffer (in number of slots)
     */
    public int getBufferSize() {
        return mBufferSize;
    }

    /**
     * Set the current maximum size of the buffer (in number of slots)
     * 
     * @param iSize new number of available slots in the buffer
     */
    public void setBufferSize(int iSize) {
        if (iSize > 10) {
            mBufferSize = iSize;
        }
    }

    /**
     * Reset all information stored into that buffer
     */
    public void resetBuffer() {
        mKeyValues.clear();
        mKeyUsageWeigth.clear();
    }

    /**
     * Returns the buffered nodes' collection for the given path
     * 
     * @param key where the nodes are requested
     * @return node collection (or null if not found)
     */
    public Collection<V> get(U key) {
    	Collection<V> result = null;
    	if( mKeyValues.containsKey(key) ) {
    		Integer cost = mKeyUsageWeigth.get(key);
    		mKeyUsageWeigth.put(key, cost + 1);
            result = mKeyValues.get(key);
        }
        return result;
    }

    /**
     * Update the buffer with the given node
     * 
     * @param key where the object belongs to
     * @param value to store
     * @param time spent to get the object
     */
    public void push(U key, V value, int time) {
    	Collection<V> tmpSet = mKeyValues.get(key);
        if (tmpSet == null) {
        	tmpSet = getEmptyCollection();
            tmpSet.add( value );
            put(key, tmpSet, 1);
        }
        else {
            tmpSet.add( value );
        }
    }

    /**
     * Stores given nodes in the buffer for the given path.
     * 
     * @param key the nodes belong to
     * @param values found at the given path
     * @param time spent to get the list of children
     * @see getEmptyCollection
     */
    public void push(U key, Collection<V> values, int time) {
    	Collection<V> tmpSet = mKeyValues.get(key);
        if (tmpSet == null) {
    		tmpSet = getEmptyCollection();
        }
        tmpSet.addAll(values);
        put(key, tmpSet, time);
    }
    
    /**
     * Returns a new collection that can be used of the default type used by the Buffer,
     * it instantiates it with right Collator if any. 
     */
    public Collection<V> getEmptyCollection() {
    	Collection<V> result;
    	if( mCollator != null ) {
    		result = new TreeSet<V> (mCollator);
    	}
    	else {
    		result = new ArrayList<V>();
    	}
        return result;
    }

    /**
     * Returns the currently used collator to sort buffer.
     * @return
     */
    public Comparator<V> getCollator() {
    	return mCollator;
    }
    
    /**
     * Set the collator to be used when storing object.
     */
    public void setCollator(Comparator<V> collator) {
    	mCollator = collator;
    }

    // ---------------------------------------------------------
    // ---------------------------------------------------------
    // Private methods
    // ---------------------------------------------------------
    // ---------------------------------------------------------
    private void put(U key, Collection<V> values, int iTimeToAccessNode) {
        freeBufferSpace();

        Integer cost = mKeyUsageWeigth.get(key);
        if (cost == null) {
            cost = iTimeToAccessNode;
        }
        else {
            cost += iTimeToAccessNode + 1;
        }
        mKeyUsageWeigth.put(key, cost);
        mKeyValues.put(key, values);
    }

    private void freeBufferSpace() {
        if (mKeyValues.size() > mBufferSize && mBufferSize > 0) {
            int iNumToRemove = (mBufferSize / 2), iRemovedItem = 0, iInfLimit;
            Object[] frequency = mKeyUsageWeigth.values().toArray();
            java.util.Arrays.sort(frequency);
            iInfLimit = (Integer) frequency[frequency.length / 2];
            Iterator<U> keys_iter = mKeyUsageWeigth.keySet().iterator();
            int freq;
            U key;
            while (keys_iter.hasNext() && iRemovedItem < iNumToRemove) {
                key = keys_iter.next();
                freq = mKeyUsageWeigth.get(key);

                if (freq <= iInfLimit) {
                    keys_iter.remove();
                    mKeyValues.remove(key);
                    iRemovedItem++;
                }
            }
        }
    }
}
