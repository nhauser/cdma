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

import org.gumtree.data.dictionary.IContext;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IKey;

public class Context implements IContext {
	private IDataset    m_dataset;
	private IContainer  m_caller;
	private IKey        m_key;
	private IPath       m_path;
	private Object[]    m_params;
	
	public Context(IDataset dataset) {
		m_dataset = dataset;
	}
	
	public Context( IDataset dataset, IContainer caller, IKey key, IPath path ) {
		m_dataset = dataset;
		m_caller  = caller;
		m_key     = key;
		m_path    = path;
	}
	
	@Override
	public String getFactoryName() {
		return m_dataset.getFactoryName();
	}

	@Override
	public IDataset getDataset() {
		return m_dataset;
	}

	@Override
	public void setDataset(IDataset dataset) {
		m_dataset = dataset;
	}

	@Override
	public IContainer getCaller() {
		return m_caller;
	}

	@Override
	public void setCaller(IContainer caller) {
		m_caller = caller;
	}

	@Override
	public IKey getKey() {
		return m_key;
	}

	@Override
	public void setKey(IKey key) {
		m_key = key;
	}

	@Override
	public IPath getPath() {
		return m_path;
	}

	@Override
	public void setPath(IPath path) {
		m_path = path;
	}

	@Override
	public Object[] getParams() {
		return m_params;
	}

	@Override
	public void setParams(Object[] params) {
		m_params = params;
	}
}
