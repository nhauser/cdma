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
package org.gumtree.data.dictionary.impl;

/// @cond internal

/**
 * The IClassLoader aims to provide a mechanism permitting to dynamically
 * load a class that is needed by the plug-in. And then invoke a method belonging
 * to that class.
 * Sometimes data need to be processed before returning it to the CDMA interfaces.
 * Such data will be managed by an external class in a package. The implementing
 * IClassLoader must be able to retrieve and execute that class which is plug-in dependent.
 */

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
import org.gumtree.data.exception.NoResultException;

public final class ExternalClassLoader extends URLClassLoader implements IClassLoader {
    public ExternalClassLoader(String factoryName, String version) {
        super(new URL[] {} );
        mFactory = factoryName;
        mVersion = version;
        mLoaded  = new HashMap<String, Class<?> >();
    } 

    /**
     * Execute the method that is given using it's namespace. The corresponding
     * class will be searched, loaded and instantiated, so the method can be called.
     * 
     * @param methodNameSpace full namespace of the method (package + class + method name)
     * @param context of the CDMA status while invoking the so called method
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
        boolean found = false;
        for( Method meth : c.getMethods() ) {
            if( meth.getName().equals(methodName) ) {
                found = true;
                try {
                    result = meth.invoke( c.newInstance(), context );
                } catch (InvocationTargetException e) {
                    throw new InvocationTargetException(e, "Error occured while invoking method: " + methodNameSpace);
                }

                break;
            }
        }
        if( !found ) {
            throw new NoResultException("Method not found in class path!");
        }
        return result;
    }

    @Override
    protected Class<?> findClass(String name) {
        Class<?> result = null;

        // Has this class been already loaded
        if( mLoaded.containsKey(name)) {
            result = mLoaded.get(name);
        }
        else {
            try {
                // Extract the package name
                String namespace = name.replaceAll("((.*[^\\.])+\\.)?([^\\.]+$)+", "$2");

                // Get folder containing the package
                File path = new File(Factory.getMappingDictionaryFolder( Factory.getFactory(mFactory) ));

                // Construct an URL
                try {
                    URL url = new URL("file", "", path.getAbsolutePath() + '/' + mVersion + '/' + namespace + ".jar" );
                    this.addURL(url);
                } catch (MalformedURLException e) {
                }

                // Ask the class loader to load the class
                result = super.findClass(name);
                mLoaded.put(name, result);
            } catch (ClassNotFoundException e) {
            }
        }
        return result;
    }

    @Override
    public String getFactoryName() {
        return mFactory;
    }
    
    // private members
    private String mVersion; // plugin implementation version
    private String mFactory; // plugin implementation factory

    private Map<String, Class<?> > mLoaded; // Class name / Class already loaded
}

/// @endcond internal