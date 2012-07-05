/*******************************************************************************
 * Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 ******************************************************************************/
package org.gumtree.data.interfaces;

import org.gumtree.data.exception.InvalidRangeException;

/**
 * Iterator for slicing an Array. 
 * This is a way to iterate over slices of arrays, and should eventually be
 * incorporated into the Gumtree Data Model rather than lying around here. Each
 * iteration returns an array of dimension dim, representing the last dim
 * dimensions of the input array. So for 3D data consisting of a set of 2D
 * arrays, each of the 2D arrays will be returned.
 *
 * @author nxi
 * 
 */
public interface ISliceIterator extends IModelObject {

	/**
	 * Check if there is next slice.
	 * 
	 * @return Boolean type Created on 10/11/2008
	 */
	boolean hasNext();

	/**
	 * Jump to the next slice.
	 * 
	 * Created on 10/11/2008
	 */
	void next();

	/**
	 * Get the next slice of Array.
	 * 
	 * @return GDM Array
	 * @throws InvalidRangeException
	 *             Created on 10/11/2008
	 */
	IArray getArrayNext() throws InvalidRangeException;

	/**
	 * Get the current slice of Array.
	 * 
	 * @return GDM Array
	 * @throws InvalidRangeException
	 *             Created on 10/11/2008
	 */
	IArray getArrayCurrent() throws InvalidRangeException;

	/**
	 * Get the shape of any slice that is returned. This could be used when a
	 * temporary array of the right shape needs to be created.
	 * 
	 * @return dimensions of a single slice from the iterator
	 * @throws InvalidRangeException
	 *             invalid range
	 */
	int[] getSliceShape() throws InvalidRangeException;
	
	/**
	 * Get the slice position in the whole array from which this slice iterator
	 * was created.
	 * @return <code>int</code> array of the current position of the slice
	 * @note rank of the returned position is the same as the IArray shape we are slicing 
	 */
	public int[] getSlicePosition();
}
