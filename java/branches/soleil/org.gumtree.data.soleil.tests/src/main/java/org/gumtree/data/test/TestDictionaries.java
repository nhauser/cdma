package org.gumtree.data.test;

import java.io.IOException;
import java.util.List;

import org.gumtree.data.Factory;
import org.gumtree.data.dictionary.IExtendedDictionary;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IKey;

public class TestDictionaries {
	public static void testKeys(ILogicalGroup group) {
		testKeys(group, 0);
	}
	
	public static void testKeys(ILogicalGroup group, int depth) {
    	long start = 0;
    	if( depth == 0 ) {
    		start = System.currentTimeMillis();
    		System.out.println("------------------------------------------------------------------");
    		System.out.println("------------------------------------------------------------------");
    		System.out.println("Dictionary structure:" + Factory.getActiveView() );
    		System.out.println("--> root = " + Factory.getKeyDictionaryPath());
    	}
    	if( group == null ) {
    		return;
    	}
    	String prefix = "", prefix2 = "";
    	for( int i = 0; i < depth; i++ ) {
    		prefix  += "|  ";
    		prefix2 += "|  ";
    		
    	}
    	
    	prefix  += "|--->";
    	prefix2 += "|--->";
    	
    	IExtendedDictionary dict = (IExtendedDictionary) group.getDictionary();
    	if( depth == 0 ) {
    		System.out.println("--> mapping = " + dict.getMappingFilePath());
    	}
    	List<IDataItem> items = null;
    	ILogicalGroup grp = null;
    	for( IKey key : dict.getAllKeys() ) {
    		items = group.getDataItemList(key);
    		for( IDataItem item : items) {
    			System.out.println(prefix + " " + key);
    		}
    		grp = group.getGroup(key);
    		if( grp != null ) {
    			System.out.println(prefix2 +" "+ key);
    			testKeys( grp, depth + 1 );
    		}
    		if( grp == null && items.isEmpty() ) {
    			if( dict.getPath(key) == null ) {
    				System.out.println(prefix + " " + key + " => Path in mapping = NULL ");
    			}
    			else {
    				System.out.println(prefix + " " + key + " => Value in file = NULL ");
    			}
    		}
    	}
    	if( depth == 0 ) {
    		System.out.println("testKeys = " + (System.currentTimeMillis() - start) / 1000.  + "s" );
    	}
    }
    
	public static void testValues(ILogicalGroup group) {
		testValues(group, 0);
	}
	
    public static void testValues(ILogicalGroup group, int depth) {
    	long start = 0;
    	if( depth == 0 ) {
    		System.out.println("------------------------------------------------------------------");
    		System.out.println("------------------------------------------------------------------");
    		System.out.println("File content:");
    		start = System.currentTimeMillis();
    	}
    	if( group == null ) {
    		return;
    	}

    	IDictionary dict = group.getDictionary();
    	List<IDataItem> items = null;
    	ILogicalGroup grp = null;
    	for( IKey key : dict.getAllKeys() ) {
    		items = group.getDataItemList(key);
    		for( IDataItem item : items) {
    			System.out.println("----------------------------");
    			System.out.println("Key: " + key);
    			System.out.println("Item: " + item.toStringDebug());
    			
    		}
    		grp = group.getGroup(key);
    		if( grp != null ) {
    			System.out.println("----------------------------");
    			System.out.println("Group: "+ key);
    			testValues( grp, depth + 1 );
    		}    		
    	}
    	if( depth == 0 ) {
    		System.out.println("testValues = " + (System.currentTimeMillis() - start) / 1000.  + "s" );
    	}
    }
}
