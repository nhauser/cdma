package org.gumtree.data.engine.jnexus;

import java.io.File;
import java.io.FilenameFilter;
import java.net.URI;

import org.gumtree.data.IDatasource;

public class NexusDatasource implements IDatasource {
	
	static public class NeXusFilter implements FilenameFilter {
	    public boolean accept(File dir, String name) {
	        return (name.endsWith(".nxs"));
	    }
	}
	
	@Override
	public String getFactoryName() {
		return NexusFactory.NAME;
	}

	@Override
	public boolean isReadable(URI target) {
		File file      = new File(target);
		String name    = file.getName();
		int length     = name.length();
		boolean result = name.substring(length - 4).equals(".nxs");
		
		return result;
	}

	@Override
	public boolean isProducer(URI target) {
		return false;
	}

	@Override
	public boolean isExperiment(URI target) {
		return false;
	}

	@Override
	public boolean isBrowsable(URI target) {
		return false;
	}
}
