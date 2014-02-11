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
package org.cdma.plugin.xml;

import java.io.File;
import java.util.Hashtable;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class XmlUtils {

    /**
     * @param file
     * @return
     * @throws Exception
     *             8 juil. 2005
     */
    public static Node getRootNode(File file) throws Exception {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

            // should we add dtd validating?
            factory.setValidating(false);
            factory.setIgnoringComments(true);

            DocumentBuilder builder = factory.newDocumentBuilder();

            // builder.setErrorHandler ( error );
            Document doc = builder.parse(file);

            return doc.getDocumentElement();
        } catch (Exception e) {
            throw e;
            // throw e;
        }
    }

    /**
     * @param noeudATester
     * @return 8 juil. 2005
     */
    public static boolean isAFakeNode(Node noeudATester) {
        int typeNode = noeudATester.getNodeType();
        boolean result = (typeNode != Node.ELEMENT_NODE);
        // Consider text node with data
        if (Node.TEXT_NODE == typeNode) {
            String value = noeudATester.getNodeValue();
            if (value != null) {
                value = value.trim();
                result = value.isEmpty();
            }
        }
        return result;

    }

    /**
     * @param nodeToTest
     * @return 8 juil. 2005
     */
    public static boolean hasRealChildNodes(Node nodeToTest) {
        boolean ret = false;

        NodeList potentialChilds = nodeToTest.getChildNodes();
        for (int i = 0; i < potentialChilds.getLength(); i++) {
            Node nextNode = potentialChilds.item(i);
            if (!isAFakeNode(nextNode)) {
                return true;
            }
        }

        return ret;
    }

    /**
     * @param noeudDOM
     * @return
     * @throws Exception
     *             8 juil. 2005
     */
    public static Hashtable<String, String> loadAttributes(Node DOMnode) throws Exception {
        Hashtable<String, String> retour = new Hashtable<String, String>();
        if (DOMnode.hasAttributes()) {
            NamedNodeMap listAtts = DOMnode.getAttributes();

            for (int i = 0; i < listAtts.getLength(); i++) {
                String nomAtt = listAtts.item(i).getNodeName().trim();
                String valueAtt = listAtts.item(i).getNodeValue().trim();
                retour.put(nomAtt, valueAtt);
            }
        }

        return retour;
    }

}
