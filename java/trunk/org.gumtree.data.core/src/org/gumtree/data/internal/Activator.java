package org.gumtree.data.internal;

import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

	private static Activator instance;
	
	private BundleContext context;
	
	public void start(BundleContext context) throws Exception {
		this.context = context;
		instance = this;
	}

	public void stop(BundleContext context) throws Exception {
		instance = null;
		this.context = null;
	}

	public BundleContext getContext() {
		return context;
	}
	
	public static Activator getDefault() {
		return instance;
	}
	
}
