package org.cdma.internal.dictionary.readers;

import java.util.HashMap;
import java.util.Map;

import org.cdma.internal.dictionary.solvers.ItemSolver;

public class DataMapping implements Cloneable {
	private String mVersion;
	private Map<String, ItemSolver> mSolvers;
	
	public DataMapping( String version ) {
		mVersion = version;
		mSolvers = new HashMap<String,ItemSolver>();
	}

	public void addSolver(String keyID, ItemSolver itemSolver) {
		mSolvers.put(keyID, itemSolver);
	}
	
	public DataMapping clone() throws CloneNotSupportedException {
		DataMapping clone = new DataMapping(mVersion);
		clone.mSolvers = new HashMap<String, ItemSolver>(mSolvers);
		return clone;
	}
	
	public Map<String, ItemSolver> getSolvers() {
		return mSolvers;
	}
}
