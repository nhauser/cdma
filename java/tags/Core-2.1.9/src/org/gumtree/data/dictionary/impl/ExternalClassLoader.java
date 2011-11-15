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

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.HashMap;
import java.util.Map;

import org.gumtree.data.Factory;
import org.gumtree.data.dictionary.IClassLoader;
import org.gumtree.data.dictionary.IContext;

public class ExternalClassLoader extends URLClassLoader implements IClassLoader {
	String m_version;
	String m_factory;
	
	Map<String, Class<?> > m_loaded;
	
	public ExternalClassLoader(String factoryName, String version) {
		super(new URL[] {} );
		m_factory = factoryName;
		m_version = version;
		m_loaded  = new HashMap<String, Class<?> >();
	} 
	
    /**
     * Execute the method that is given using it's namespace. The corresponding
     * class will be searched, loaded and instantiated, so the method can called.
     * 
     * @param methodNameSpace full namespace of the method
     * @param source the CDM object that has requested this invocation
     * @param args 
     * @return List of IObject that have been created using the called method.
     * @throws Exception in case of any trouble
     * 
     * @note the method's namespace must be that form: my.package.if.any.MyClass.MyMethod
     */
	@Override
    public Object invoke( String methodNameSpace, IContext context ) throws Exception {
    	Object result = null;
    	
    	// Extract package, class and method names
    	String className  = methodNameSpace.replaceAll("(^.*[^\\.]+)(\\.[^\\.]+$)+", "$1");
    	String methodName = methodNameSpace.replaceAll("(^.*[^\\.]+\\.)([^\\.]+$)+", "$2");
    	
    	// Load the class
		Class<?> c = findClass(className);
		for( Method meth : c.getMethods() ) {
			if( meth.getName().equals(methodName) ) {
				try {
					result = meth.invoke( c.newInstance(), context );
				} catch (InvocationTargetException e) {
					throw new InvocationTargetException(e, "Error occured while invoking method: " + methodNameSpace);
				}
				
				break;
			}
		}
    	return result;
    }

	@Override
    protected Class<?> findClass(String name) {
        Class<?> result = null;
        
        // Has this class been already loaded
        if( m_loaded.containsKey(name)) {
        	result = m_loaded.get(name);
        }
        else {
			try {
				// Extract the package name
				String namespace = name.replaceAll("((.*[^\\.])+\\.)?([^\\.]+$)+", "$2");
		   
				// Get folder containing the package
		    	File path = new File(Factory.getMappingDictionaryFolder( Factory.getFactory(m_factory) ));
				
				// Construct an URL
		    	try {
		    		URL url = new URL("file", "", path.getAbsolutePath() + '/' + m_version + '/' + namespace + ".jar" );
					this.addURL(url);
				} catch (MalformedURLException e) {
					e.printStackTrace();
				}
				
				// Ask the class loader to load the class
				result = super.findClass(name);
				m_loaded.put(name, result);
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
        }
        return result;
    }

	@Override
	public String getFactoryName() {
		return m_factory;
	}
}
