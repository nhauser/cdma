package org.gumtree.data.dictionary;

import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.interfaces.IModelObject;


/**
 * This interface is used when invoking an external method.
 * It should contain all required information so the called method,
 * can work properly as if it were in the CDM.
 * The context is compound of the dataset we are working on,
 * the caller of the method, the key used to call that method (that can
 * have some parameters), the path (with parameters set) and some
 * parameters that are set by the institute's plug-in.
 */
public interface IContext extends IModelObject {
	/**
	 * Permits to get the IDataset we want to work on.
	 */
	public IDataset getDataset();
	public void     setDataset(IDataset dataset);
	/**
	 * Permits to get the IContainer that instantiated the context.
	 */
	public IContainer  getCaller();
	public void     setCaller(IContainer caller);
	
	/**
	 * Permits to get the IKey that lead to this instantiation.
	 */
	public IKey     getKey();
	public void     setKey(IKey key);
	
	/**
	 * Permits to get the IPath corresponding to the IKey
	 */
	public IPath    getPath();
	public void     setPath(IPath path);
	
	/**
	 * Permits to have some parameters that are defined by the instantiating plug-in
	 * and that can be useful for the method using this context.
	 *  
	 * @return array of object
	 */
	public Object[] getParams();
	public void     setParams(Object[] params);
}