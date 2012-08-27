//****************************************************************************
// Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Clément RODRIGUEZ (clement.rodriguez@synchroton-soleil.fr) - initial API and implementation
// ****************************************************************************
package org.cdma.interfaces;

import java.net.URI;
import java.util.List;

import org.cdma.internal.IModelObject;

/// @cond pluginAPIclientAPI

/**
 * @brief IDatasource is used by the main Factory to detect plug-in compatibility.
 * 
 * The main focus of this interface is to determine, which plug-in can read the URI 
 * and if possible to which one it belongs to.
 * <p>
 * It distinguish mostly two things: that the plug-in can read the given URI and if
 * the target was written using that plug-in. That last point means that Extended Dictionary
 * mechanism can be used.
 * <p>
 * It can also be used, by the main Factory, to discover if a specific targeted URI
 * is considered by plug-ins as a DATASET or a simple FOLDER. More particularly if
 * the URI interpreted by this plug-in's instance can be browsed to find a IDataset.
 * 
 * @author rodriguez
 */
public interface IDatasource extends IModelObject 
{
    /**
     * Returns true if the target has a compatible data format with that plug-in instance.
     * Can that plug-in read the targeted data source ?
     * 
     * @param target of the data source can be a file, a folder, data base...
     * @return true if the target is compatible
     */
    boolean isReadable( URI target );

    /**
     * Returns true if the given URI aims to a data source that is compatible with that 
     * plug-in's instance AND if the data source was produced by the plug-in.
     * 
     * @param target of the data source can be a file, a folder, data base...
     * @return true if the target is fully compatible with the plug-in
     */
    boolean isProducer( URI target );

    /**
     * Returns true if the target can be browsed with that plug-in. The aim is to determine
     * the entry point of a data source (i.e to obtain a IDataset that provides 
     * the Extended Dictionary mechanism). If the target points an experiment for that plug-in
     * (in the meaning of the Extended Dictionary mechanism) it will return false.
     * 
     * @param target of the data source can be a file, a folder, data base, group or data item in a data file...
     * @return true if the target can be browsed until it points on a IDataset path for that plug-in's instance
     * @note a first application of this method is a dialog box for opening a IDataset from an URI where several
     * data sources of different formats are stored.
     * @see IDatasource#isExperiment(URI)
     */
    boolean isBrowsable( URI target );

    /**
     * Returns true if the target is considered as an experiment by that plug-in. The aim 
     * is to determine the entry point of a that data source (i.e to obtain a IDataset that
     * provides the Extended Dictionary mechanism). If the target points an experiment for 
     * that plug-in (in the meaning of the Extended Dictionary mechanism) it will return true.
     * 
     * @param target of the data source can be a group or data item within a file, a folder, data base ...
     * @return true if the target can't be browsed anymore because it aims IDataset path for that plug-in's instance
     * @note a first application of this method is a dialog box for opening a IDataset from an URI where several
     * data sources of different formats are stored.
     * @see IDatasource#isBrowsable(URI)
     */
    boolean isExperiment( URI target );
    
    
    /**
     * Returns the list of available sub URIs. That method has a meaning only when
     * isBrowsable returns true and isExperiment is false.
     * 
     * The aim is to obtain the list of sub available URI to iterate again on it and 
     * discover in the file system (that might be proper to the plug-in) the experiment
     * entry point. 
     * 
     * @param target
     * @return list of URI whom elements have a meaning for this plug-in.
     */
    List<URI> getValidURI( URI target );
    
    /**
     * Returns an array of string compound of each element or the URI
     * but the protocol.
     */
    String[] getURIParts( URI target );

    
    /**
     * Return the last modification of the given URI if it can be managed by 
     * this plug-in. Else return 0.
     * 
     * @param URI target to obtain last modification date
     * @return long representing the last modification
     */
    long getLastModificationDate( URI target );
}

/// @endcond pluginAPIclientAPI