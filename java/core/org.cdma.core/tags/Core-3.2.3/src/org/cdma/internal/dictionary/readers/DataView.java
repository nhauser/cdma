package org.cdma.internal.dictionary.readers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class DataView implements Cloneable {
	private List<String>          mKeyItem;    
//	private List<String>          mKeyView; 
//	private Map<String, String>   mLink;
	private Map<String, DataView> mViews;
	private List<String>          mAllKeys;
	private String                mName;
	
	public DataView(String name) {
		this();
		mName = name;
	}
	
	public DataView() {
//		mLink    = new HashMap<String, String>();
		mViews   = new HashMap<String, DataView>();
		mKeyItem = new ArrayList<String>();
		mAllKeys = new ArrayList<String>();
		mName    = "";
//		mKeyView = new ArrayList<String>();
	}
	
	public void addKey( String key ) {
		if( ! mAllKeys.contains( key ) ) {
			mKeyItem.add( key );
			mAllKeys.add( key );
		}
	}
	
	public void addView( String key, DataView view ) {
//		mKeyView.add(key);
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
