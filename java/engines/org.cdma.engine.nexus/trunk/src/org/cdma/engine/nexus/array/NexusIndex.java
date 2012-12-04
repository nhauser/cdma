//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.nexus.array;

import java.util.List;

import org.cdma.interfaces.IIndex;
import org.cdma.interfaces.IRange;
import org.cdma.utilities.memory.DefaultIndex;

public final class NexusIndex extends DefaultIndex implements Cloneable {
    /// Constructors
    public NexusIndex(String factoryName, fr.soleil.nexus.DataItem ds) {
        super(factoryName, ds.getSize(), new int[ds.getSize().length], ds.getSize());
    }

    public NexusIndex(String factoryName, int[] shape) {
        super(factoryName, shape.clone(), new int[shape.length], shape.clone());
    }

    public NexusIndex(NexusIndex index) {
        super(index);
    }

    public NexusIndex(String factoryName, int[] shape, int[] start, int[] length) {
        super( factoryName, shape, start, length );
    }

    public NexusIndex(String factoryName, List<IRange> ranges) {
        super(factoryName, ranges);
    }

    @Override
    public IIndex clone() {
        return new NexusIndex(this);
    }
}
