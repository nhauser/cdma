package org.gumtree.data.engine.jnexus;

import java.io.File;
import java.io.FilenameFilter;
import java.net.URI;

import org.gumtree.data.IDatasource;

public final class NexusDatasource implements IDatasource {
	
	public static final class NeXusFilter implements FilenameFilter {
		public static final int    EXTENSION_LENGTH = 4;
		public static final String EXTENSION = ".nxs";
		
	    public boolean accept(File dir, String name) {
	        return (name.endsWith(NeXusFilter.EXTENSION));
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
		boolean result = length > NeXusFilter.EXTENSION_LENGTH && name.substring(length - NeXusFilter.EXTENSION_LENGTH).equals(NeXusFilter.EXTENSION);
		
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
