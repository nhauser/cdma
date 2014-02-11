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
package org.cdma.internal.dictionary.readers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class DataView implements Cloneable {
	private List<String>          mKeyItem;    
	private Map<String, DataView> mViews;
	private List<String>          mAllKeys;
	private String                mName;
	
	public DataView(String name) {
		this();
		mName = name;
	}
	
	public DataView() {
		mViews   = new HashMap<String, DataView>();
		mKeyItem = new ArrayList<String>();
		mAllKeys = new ArrayList<String>();
		mName    = "";
	}
	
	public void addKey( String key ) {
		if( ! mAllKeys.contains( key ) ) {
			mKeyItem.add( key );
			mAllKeys.add( key );
		}
	}
	
	public void addView( String key, DataView view ) {
		if( ! mAllKeys.contains( key ) ) {
			mViews.put( key, view );
			mAllKeys.add( key );
		}
	}

	public DataView getView(String keyID) {
		DataView view;
		
		if( mViews.containsKey(keyID) ) {
			view = mViews.get(keyID);
		}
		else {
			view = new DataView();
		}
		return view;
	}
	
	public List<String> getItemKeys() {
		return mKeyItem;
	}
	
	public List<String> getAllKeys() {
		return mAllKeys;
	}
	
	public Map<String, DataView> getSubViews() {
		return mViews;
	}
	
	public boolean containsKey(String keyName) {
		return mAllKeys.contains(keyName);
	}
	
	public DataView clone() throws CloneNotSupportedException {
		DataView clone = new DataView();
		clone.mViews   = new HashMap<String, DataView>(mViews);
		clone.mKeyItem = new ArrayList<String>(mKeyItem);
		clone.mAllKeys = new ArrayList<String>(mAllKeys);
		return clone;
	}

	public String getName() {
		return mName;
	}
	
	public void setName( String name) {
		mName = name;
	}
}
