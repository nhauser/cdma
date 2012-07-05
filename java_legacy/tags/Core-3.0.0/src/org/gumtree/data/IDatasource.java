 /** ****************************************************************************
 * Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Clément RODRIGUEZ (clement.rodriguez@synchroton-soleil.fr) - initial API and implementation
 **************************************************************************** **/
package org.gumtree.data;

import java.net.URI;
import org.gumtree.data.interfaces.IModelObject;

/**
 * The main focus of this interface is to determine, which plug-in can read the URI 
 * and if possible to which one it belongs to.
 * <p>
 * It is is used, by the main Factory, to discover if a specific targeted URI
 * is considered by plug-ins as a DATASET or a simple FOLDER. The Factory will interrogate
 * each plug-in.
 * 
 * @author rodriguez
 */
public interface IDatasource extends IModelObject {

	boolean isReadable( URI target ); // permet d'avoir un dataset navigable  "file://$home/mon_fichier.nxs/"

	boolean isProducer( URI target ); // permet d'isoler un plugin  "file://$home/mon_fichier.nxs/"

	boolean isExperiment( URI target ); // on a reconnu une manipe avec laquelle on pourra utiliser un dico "plugin://$home/mon_fichier.nxs/entry/experiment"

	boolean isBrowsable( URI target );  // on a reconnu un dataset navigable ET dans lequel on peut eventulklement obtnir une ou des 'experiments' "plugin://$home/mon_fichier.nxs/entry/experiment"

}
