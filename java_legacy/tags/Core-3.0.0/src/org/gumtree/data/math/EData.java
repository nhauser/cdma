/*******************************************************************************
 * Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 ******************************************************************************/
package org.gumtree.data.math;

/**
 * The data wrapper for error propagation mathematics.
 * 
 * @param <T>
 *            the parameter class type
 * @author nxi Created on 09/07/2008
 */
public class EData<T> {

	/**
	 * Data storage.
	 */
	private T data;
	/**
	 * Variance data storage.
	 */
	private T variance;

	/**
	 * Constructor fully parameterised.
	 * 
	 * @param data
	 *            T type
	 * @param variance
	 *            T type
	 */
	public EData(final T data, final T variance) {
		this.data = data;
		this.variance = variance;
	}

	/**
	 * Return the data field.
	 * 
	 * @return a generic type Created on 15/07/2008
	 */
	public T getData() {
		return data;
	}

	/**
	 * Return the variance field.
	 * 
	 * @return a generic type Created on 15/07/2008
	 */
	public T getVariance() {
		return variance;
	}
}
