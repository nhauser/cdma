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
package org.gumtree.data.impl.netcdf;

import org.gumtree.data.impl.NcFactory;
import org.gumtree.data.interfaces.IArrayIterator;

import ucar.ma2.IndexIterator;

/**
 * An iterator for GDM IArray object.
 * 
 * @author nxi
 * 
 */
public class NcArrayIterator implements IArrayIterator {

	/**
	 * Wrapper of Netcdf IndexIterator.
	 */
	private IndexIterator iterator = null;

	/**
	 * Wrapper constructor.
	 * 
	 * @param indexIterator
	 *            Netcdf IndexIterator
	 */
	public NcArrayIterator(final IndexIterator indexIterator) {
		this.iterator = indexIterator;
	}

	@Override
	public boolean getBooleanCurrent() {
		return iterator.getBooleanCurrent();
	}

	@Override
	public boolean getBooleanNext() {
		return iterator.getBooleanNext();
	}

	@Override
	public byte getByteCurrent() {
		return iterator.getByteCurrent();
	}

	@Override
	public byte getByteNext() {
		return iterator.getByteNext();
	}

	@Override
	public char getCharCurrent() {
		return iterator.getCharCurrent();
	}

	@Override
	public char getCharNext() {
		return iterator.getCharNext();
	}

	@Override
	public int[] getCurrentCounter() {
		return iterator.getCurrentCounter();
	}

	@Override
	public double getDoubleCurrent() {
		return iterator.getDoubleCurrent();
	}

	@Override
	public double getDoubleNext() {
		return iterator.getDoubleNext();
	}

	@Override
	public float getFloatCurrent() {
		return iterator.getFloatCurrent();
	}

	@Override
	public float getFloatNext() {
		return iterator.getFloatNext();
	}

	@Override
	public int getIntCurrent() {
		return iterator.getIntCurrent();
	}

	@Override
	public int getIntNext() {
		return iterator.getIntNext();
	}

	@Override
	public long getLongCurrent() {
		return iterator.getLongCurrent();
	}

	@Override
	public long getLongNext() {
		return iterator.getLongNext();
	}

	@Override
	public Object getObjectCurrent() {
		return iterator.getObjectCurrent();
	}

	@Override
	public Object getObjectNext() {
		return iterator.getObjectNext();
	}

	@Override
	public short getShortCurrent() {
		return iterator.getShortCurrent();
	}

	@Override
	public short getShortNext() {
		return iterator.getShortNext();
	}

	@Override
	public boolean hasNext() {
		return iterator.hasNext();
	}

	@Override
	public Object next() {
		return iterator.next();
	}

	@Override
	public void setBooleanCurrent(final boolean val) {
		iterator.setBooleanCurrent(val);
	}

	@Override
	public void setBooleanNext(final boolean val) {
		iterator.setBooleanNext(val);
	}

	@Override
	public void setByteCurrent(final byte val) {
		iterator.setByteCurrent(val);
	}

	@Override
	public void setByteNext(final byte val) {
		iterator.setByteNext(val);
	}

	@Override
	public void setCharCurrent(final char val) {
		iterator.setCharCurrent(val);
	}

	@Override
	public void setCharNext(final char val) {
		iterator.setCharNext(val);
	}

	@Override
	public void setDoubleCurrent(final double val) {
		iterator.setDoubleCurrent(val);
	}

	@Override
	public void setDoubleNext(final double val) {
		iterator.setDoubleNext(val);
	}

	@Override
	public void setFloatCurrent(final float val) {
		iterator.setFloatCurrent(val);
	}

	@Override
	public void setFloatNext(final float val) {
		iterator.setFloatNext(val);
	}

	@Override
	public void setIntCurrent(final int val) {
		iterator.setIntCurrent(val);
	}

	@Override
	public void setIntNext(final int val) {
		iterator.setIntNext(val);
	}

	@Override
	public void setLongCurrent(final long val) {
		iterator.setLongCurrent(val);
	}

	@Override
	public void setLongNext(final long val) {
		iterator.setLongNext(val);
	}

	@Override
	public void setObjectCurrent(final Object val) {
		iterator.setObjectCurrent(val);
	}

	@Override
	public void setObjectNext(final Object val) {
		iterator.setObjectNext(val);
	}

	@Override
	public void setShortCurrent(final short val) {
		iterator.setShortCurrent(val);
	}

	@Override
	public void setShortNext(final short val) {
		iterator.setShortNext(val);
	}

	@Override
	public String getFactoryName() {
		return NcFactory.NAME;
	}

	@Override
	public boolean hasCurrent() {
		// TODO Auto-generated method stub
		return false;
	}

}
