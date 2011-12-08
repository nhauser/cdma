package org.gumtree.data.soleil.array;

import org.gumtree.data.engine.jnexus.array.NexusIndex;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.soleil.NxsFactory;

public class NxsIndex extends NexusIndex {
	private NexusIndex m_indexArrayData;
	private NexusIndex m_indexStorage;
	
	
	/// Constructors
	/**
	 * Construct a matrix of index. The matrix is defined by shape, start and length
	 * that also carries underlying data storage dimensions. The matrixRank defines
	 * which values of those arrays concern the matrix of index and which concern
	 * the index.
	 * The first matrixRank values concern the matrix of index.
	 */
	public NxsIndex(int matrixRank, int[] shape, int[] start, int[] length) {
		super(shape, start, length);
		m_indexArrayData = new NexusIndex(
				java.util.Arrays.copyOfRange(shape, 0, matrixRank),
				java.util.Arrays.copyOfRange(start, 0, matrixRank),
				java.util.Arrays.copyOfRange(length, 0, matrixRank)
		   );
		
		m_indexStorage = new NexusIndex(
				java.util.Arrays.copyOfRange(shape, matrixRank, shape.length),
				java.util.Arrays.copyOfRange(start, matrixRank, start.length),
				java.util.Arrays.copyOfRange(length, matrixRank, length.length)
		   );
    }
	
	public NxsIndex(int[] shape, int[] start, int[] length) {
		this(0, shape, start, length);
	}
	
	public NxsIndex(NxsIndex index) {
		super(index);
		
		m_indexArrayData = (NexusIndex) index.m_indexArrayData.clone();
		m_indexStorage = (NexusIndex) index.m_indexStorage.clone();
    }
	
	public NxsIndex(int[] storage) {
		super(storage);
		m_indexArrayData = new NexusIndex( new int[] {});
		m_indexStorage   = new NexusIndex(storage);
	}
	
	public NxsIndex(int[] matrix, int[] storage) {
		super(
				concat(matrix, storage)
		);
		
		m_indexArrayData = new NexusIndex(matrix);
		m_indexStorage   = new NexusIndex(storage);
	}
	
	public NxsIndex(int matrixRank, IIndex index) {
		this(matrixRank, index.getShape(), index.getOrigin(), index.getShape());

		m_indexArrayData = new NexusIndex(
				java.util.Arrays.copyOfRange(index.getShape(), 0, matrixRank),
				java.util.Arrays.copyOfRange(index.getOrigin(), 0, matrixRank),
				java.util.Arrays.copyOfRange(index.getShape(), 0, matrixRank)
		   );
		
		m_indexStorage = new NexusIndex(
				java.util.Arrays.copyOfRange(index.getShape(), matrixRank, index.getShape().length),
				java.util.Arrays.copyOfRange(index.getOrigin(), matrixRank, index.getOrigin().length),
				java.util.Arrays.copyOfRange(index.getShape(), matrixRank, index.getShape().length)
		   );
		this.set(index.getCurrentCounter());
	}
	
	public long currentElementMatrix() {
		return m_indexArrayData.currentElement();
	}
	
	public long currentElementStorage() {
		return m_indexStorage.currentElement();
	}
	
	public int[] getCurrentCounterMatrix() {
		return m_indexArrayData.getCurrentCounter();
	}
	
	public int[] getCurrentCounterStorage() {
		return m_indexStorage.getCurrentCounter();
	}

	@Override
	public void setOrigin(int[] origin) {
		if( origin.length != getRank() ) {
            throw new IllegalArgumentException();
		}
		super.setOrigin(origin);
		m_indexArrayData.setOrigin(
					java.util.Arrays.copyOfRange(origin, 0, m_indexArrayData.getRank())
				);
		
		m_indexStorage.setOrigin(
				java.util.Arrays.copyOfRange(origin, m_indexArrayData.getRank(), origin.length)
			);
	}


	@Override
	public void setShape(int[] shape) {
		if( shape.length != getRank() ) {
            throw new IllegalArgumentException();
		}
		super.setShape(shape);
		m_indexArrayData.setShape(
					java.util.Arrays.copyOfRange(shape, 0, m_indexArrayData.getRank())
				);
		
		m_indexStorage.setShape(
				java.util.Arrays.copyOfRange(shape, m_indexArrayData.getRank(), shape.length)
			);
	}


	@Override
	public void setStride(long[] stride) {
		if( stride.length != getRank() ) {
            throw new IllegalArgumentException();
		}
		super.setStride(stride);
		int iRank = m_indexArrayData.getRank();
		
		// Set the stride for the storage arrays
		m_indexStorage.setStride(
				java.util.Arrays.copyOfRange(stride, iRank, stride.length)
			);
		
		// Get the number of cells in storage arrays
		long[] iStride = m_indexStorage.getStride();
		long current = iStride[ 0 ] * m_indexStorage.getShape()[0];
		
		
		// Divide the stride by number of cells contained in storage arrays
		iStride = new long[iRank];
		for( int i = iRank; i > 0; i-- ) {
			iStride[i - 1] = stride[i - 1] / current;
		}
		
		m_indexArrayData.setStride(iStride);
	}

	@Override
	public IIndex set(int[] index) {
		if( index.length != getRank() ) {
            throw new IllegalArgumentException();
		}
		super.set(index);
		m_indexArrayData.set(
					java.util.Arrays.copyOfRange(index, 0, m_indexArrayData.getRank())
				);
		
		m_indexStorage.set(
				java.util.Arrays.copyOfRange(index, m_indexArrayData.getRank(), index.length)
			);
		
        return this;
	}

	@Override
	public void setDim(int dim, int value) {
		super.setDim(dim, value);
		int[] curPos = this.getCurrentCounter();
		curPos[dim] = value;
		this.set(curPos);
	}


	@Override
	public IIndex set0(int v) {
		setDim(0, v);
		return this;
	}


	@Override
	public IIndex set1(int v) {
		setDim(1, v);
		return this;
	}


	@Override
	public IIndex set2(int v) {
		setDim(2, v);
		return this;
	}


	@Override
	public IIndex set3(int v) {
		setDim(3, v);
		return this;
	}


	@Override
	public IIndex set4(int v) {
		setDim(4, v);
		return this;
	}

	@Override
	public IIndex set5(int v) {
		setDim(5, v);
		return this;
	}

	@Override
	public IIndex set6(int v) {
		setDim(6, v);
		return this;
	}

	@Override
	public IIndex set(int v0) {
		int[] curPos = this.getCurrentCounter();
		curPos[0] = v0;
		this.set(curPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1) {
		int[] curPos = this.getCurrentCounter();
		curPos[0] = v0;
		curPos[1] = v1;
		this.set(curPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2) {
		int[] curPos = this.getCurrentCounter();
		curPos[0] = v0;
		curPos[1] = v1;
		curPos[2] = v2;
		this.set(curPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2, int v3) {
		int[] curPos = this.getCurrentCounter();
		curPos[0] = v0;
		curPos[1] = v1;
		curPos[2] = v2;
		curPos[3] = v3;
		this.set(curPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2, int v3, int v4) {
		int[] curPos = this.getCurrentCounter();
		curPos[0] = v0;
		curPos[1] = v1;
		curPos[2] = v2;
		curPos[3] = v3;
		curPos[4] = v4;
		this.set(curPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2, int v3, int v4, int v5) {
		int[] curPos = this.getCurrentCounter();
		curPos[0] = v0;
		curPos[1] = v1;
		curPos[2] = v2;
		curPos[3] = v3;
		curPos[4] = v4;
		curPos[5] = v5;
		this.set(curPos);
		return this;
	}

	@Override
	public IIndex set(int v0, int v1, int v2, int v3, int v4, int v5, int v6) {
		int[] curPos = this.getCurrentCounter();
		curPos[0] = v0;
		curPos[1] = v1;
		curPos[2] = v2;
		curPos[3] = v3;
		curPos[4] = v4;
		curPos[5] = v5;
		curPos[6] = v6;
		this.set(curPos);
		return this;
	}

	@Override
	public void setIndexName(int dim, String indexName) {
		super.setIndexName(dim, indexName);
		if( dim >= m_indexArrayData.getRank() ) {
			m_indexStorage.setIndexName(dim, indexName);
		}
		else {
			m_indexArrayData.setIndexName(dim, indexName);
		}
	}


	@Override
	public String getIndexName(int dim) {
		return super.getIndexName(dim);
	}

	@Override
	public IIndex reduce() {
		super.reduce();
		m_indexArrayData.reduce();
		m_indexStorage.reduce();
		return this;
	}

	@Override
	public IIndex reduce(int dim) throws IllegalArgumentException {
		super.reduce(dim);
		if( dim < m_indexArrayData.getRank() ) {
			m_indexArrayData.reduce(dim);
		}
		else {
			m_indexStorage.reduce(dim - m_indexArrayData.getRank());
		}
		return this;
	}
	
	@Override
	public String toString() {
		return super.toString()+ "\n" +m_indexArrayData.toString() + "\n" + m_indexStorage;
	}
	
    @Override
    public IIndex clone() {
    	NxsIndex index = new NxsIndex(this); 
    	return index;
    }
	
	static public int[] concat(int[] array1, int[] array2) {
		int[] result = new int[array1.length + array2.length];
		System.arraycopy(array1, 0, result, 0, array1.length);
		System.arraycopy(array2, 0, result, 0, array2.length);
		return result;
	}
	
	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}
	
    // ---------------------------------------------------------
    /// Protected methods
    // ---------------------------------------------------------
	public NexusIndex getIndexMatrix() {
		return m_indexArrayData;
	}
	
	public NexusIndex getIndexStorage() {
		return m_indexStorage;
	}
}
