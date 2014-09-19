/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.utilities;

import java.util.Arrays;
import java.util.HashMap;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IArray;

/**
 * CubeData is a tool based on the CDMA API. It aims to slice a given IArray and to
 * load data by packet.
 * Constructed with a IArray and an integer that correspond to the slice's rank.
 * It will use that rank to construct a new IArray that is a portion of the first one.
 * The new IArray will correspond to latest dimension of the whole array (i.e. fastest
 * varying dimensions) starting at the given position.
 * As IArray should load data at the query, the CubeData will ask to have access to
 * the underlying data according a buffer size. The aim is to load data with an
 * optimized size.
 * For example: instead of loading 1 spectrum 300 times, it will be preferable to load
 * 30 spectrums 10 times.
 * 
 * @author rodriguez
 * 
 */

public class CubeData {
    static final int MAX_BUFFER = 20000000;
    private final int maxBuffer; // Maximum buffer size for this instance
    private final IArray mArray; // Instance of the array 'seen' by this cube
    private int mRank; // Rank of the array 'seen' by this cube
    private final int mSliceRank; // Rank of data we want to get using this cube
    private long mSliceLength; // Length of a slice
    private long mLength; // Length of the array 'seen' by this cube
    private int[] mShape; // Full shape of the array 'seen' by this cube
    private int mPos; // Current position of the lastly considered sub-cube
    private int mNbSlices; // Number of slices that compound 1 sub-cube
    private final HashMap<Integer, CubeData> mSubCubes; // sub-cubes: portion of the whole array

    public CubeData(IArray array, int sliceRank) {
        this(array, sliceRank, MAX_BUFFER);
    }

    public CubeData(IArray array, int sliceRank, int bufferSize) {
        maxBuffer = bufferSize;
        mArray = array;
        mSliceRank = sliceRank;
        mPos = 0;
        mSubCubes = new HashMap<Integer, CubeData>();

        if (mArray.getRank() == mSliceRank + 1) {
            mRank = mArray.getRank();
            mShape = mArray.getShape();
            mLength = 1;
            mSliceLength = 1;
            for (int i = 0; i < mRank; i++) {
                if (mRank - mSliceRank <= i) {
                    mSliceLength *= mShape[i];
                }
                mLength *= mShape[i];
            }

            // Calculate number of slices
            if (mLength > maxBuffer) {
                mNbSlices = (int) (maxBuffer / mSliceLength);
            } else {
                mNbSlices = 1;
            }
        }
    }

    /**
     * Get the IArray that is at the given position
     * 
     * @param position integer array of position
     * 
     * @return IArray that is a portion of the array compounding this cube
     */
    public IArray getData(int... position) {
        IArray data = null;
        int rank = mArray.getRank();
        if (rank == mSliceRank + 1) {
            if (mLength > maxBuffer) {
                // Calculate slice number of the requested position
                int posSlice = position[0] / mNbSlices;
                position[0] = position[0] % mNbSlices;

                // If sub-Cube already created
                if (mSubCubes.containsKey(posSlice)) {
                    data = mSubCubes.get(posSlice).getData(position);
                }
                // Create corresponding sub-Cube
                else {
                    // Calculate slice shape according whole shape
                    int nbSlices = mNbSlices;
                    if (posSlice * mNbSlices >= mShape[0]) {
                        nbSlices = mShape[0] % mNbSlices;
                        if (nbSlices == 0) {
                            nbSlices = mShape[0];
                        }
                    }

                    // Slice the array
                    int[] shape = Arrays.copyOf(mShape, mRank);
                    shape[0] = nbSlices;
                    int[] projPos = new int[rank];
                    projPos[0] = posSlice;

                    IArray tmpArray = null;
                    try {
                        tmpArray = mArray.getArrayUtils().section(projPos, shape).getArray();
                    } catch (InvalidRangeException e) {
                        Factory.getLogger().log(Level.SEVERE, e.getMessage());
                    }

                    CubeData tmpCube = new CubeData(tmpArray, mSliceRank, maxBuffer);
                    mSubCubes.put(posSlice, tmpCube);
                    data = mSubCubes.get(posSlice).getData(position);

                }
            } else {
                // Load the matrix
                mArray.getStorage();
                data = section(position[0]);
            }
        } else {
            if (mSubCubes.containsKey(position[0])) {
                data = mSubCubes.get(position[0]).getData(Arrays.copyOfRange(position, 1, position.length));
                mPos = position[0];
            } else {
                IArray tmpArray = section(position[0]);

                CubeData tmpCube = new CubeData(tmpArray, mSliceRank, maxBuffer);
                mSubCubes.put(position[0], tmpCube);
                mPos = position[0];
                return tmpCube.getData(Arrays.copyOfRange(position, 1, position.length));
            }

        }

        return data;
    }

    /**
     * Return the last requested position
     * 
     * @return integer array of the last requested position
     */
    public int[] getLastPos() {
        int rank = mArray.getRank() - mSliceRank;
        int[] position = new int[rank];
        if (mSubCubes.size() > 0) {
            int[] subPos = mSubCubes.get(mPos).getLastPos();
            int i = 1;
            for (int pos : subPos) {
                position[i++] = pos;
            }
        } else {
            position[0] = mPos;
        }
        return position;
    }

    // ---------------------------------------------------------
    // ---------------------------------------------------------
    // / Private methods
    // ---------------------------------------------------------
    // ---------------------------------------------------------
    private IArray section(int position) {
        IArray result = null;
        int rank = mArray.getRank();
        if (position <= mArray.getShape()[0]) {
            int[] shape = mArray.getShape();
            int[] projPos = new int[rank];

            shape[0] = 1;
            projPos[0] = position;
            try {
                result = mArray.getArrayUtils().section(projPos, shape).getArray();
            } catch (InvalidRangeException e) {
                Factory.getLogger().log(Level.WARNING, e.getMessage());
            }
        }
        return result;
    }

}
