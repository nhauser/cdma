package org.cdma.plugin.edf.navigation;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.filechooser.FileSystemView;

import org.cdma.arrays.DefaultArrayMatrix;
import org.cdma.dictionary.Path;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.SignalNotAvailableException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.interfaces.INode;
import org.cdma.plugin.edf.EdfFactory;
import org.cdma.plugin.edf.abstraction.AbstractGroup;
import org.cdma.plugin.edf.abstraction.AbstractObject;
import org.cdma.plugin.edf.utils.EdfFileReader;
import org.cdma.plugin.edf.utils.FileComparator;
import org.cdma.plugin.edf.utils.StringUtils;
import org.cdma.utils.DefaultNode;
import org.cdma.utils.DefaultPath;

public class EdfGroup extends AbstractGroup {

    protected static final String DEFAULT_SHORT_NAME = "data";

    private static final String PATH_SEPARATOR = "/";

    private final File referenceFile;
    private final ArrayList<AbstractObject> objectList;
    private final ArrayList<EdfDataItem> itemList;
    private ArrayList<EdfGroup> groupList;
    private final EdfDataset dataset;

    private boolean analyzed = false;;

    protected EdfGroup(EdfDataset dataset, File file) {
        super();
        this.dataset = dataset;
        this.referenceFile = file;
        this.groupList = new ArrayList<EdfGroup>();
        this.itemList = new ArrayList<EdfDataItem>();
        this.groupList = new ArrayList<EdfGroup>();
        this.objectList = new ArrayList<AbstractObject>();
        if (referenceFile != null) {
            setName(referenceFile.getName());
        }
        analyzeEdfFile();
        getGroupList();
    }

    public EdfGroup(File file) {
        this(null, file);
    }

    private void analyzeEdfFile() {
        if (!analyzed) {

            if (referenceFile != null) {
                HashMap<String, String> headerMap = new HashMap<String, String>();
                try {
                    FileInputStream fis = new FileInputStream(referenceFile.getAbsolutePath());
                    DataInputStream dis = new DataInputStream(fis);

                    int character = -1;

                    StringBuffer headerBuffer = new StringBuffer();
                    boolean newLine = false;
                    while (true) {
                        try {
                            character = dis.read();
                            if (character == -1) {
                                break;
                            }
                            headerBuffer.append((char) character);
                            if (character == '\n') {
                                newLine = true;
                            }
                            else {
                                if ((character == '}') && newLine) {
                                    character = dis.read();
                                    break;
                                }
                                newLine = false;
                            }
                        }
                        catch (IOException e) {
                            character = -1;
                            break;
                        }
                    }
                    if (character != -1) {
                        char toCheck = headerBuffer.charAt(0);
                        while (Character.isWhitespace(toCheck) || (toCheck == '{')) {
                            headerBuffer.delete(0, 1);
                            toCheck = headerBuffer.charAt(0);
                        }
                        toCheck = headerBuffer.charAt(headerBuffer.length() - 1);
                        while (Character.isWhitespace(toCheck) || (toCheck == '}')) {
                            headerBuffer.delete(headerBuffer.length() - 1, headerBuffer.length());
                            toCheck = headerBuffer.charAt(headerBuffer.length() - 1);
                        }
                        String[] lines = headerBuffer.toString().split(";");
                        for (String line : lines) {
                            int separatorIndex = line.lastIndexOf('=');
                            if (separatorIndex > -1) {
                                headerMap.put(line.substring(0, separatorIndex).trim(), line
                                        .substring(separatorIndex + 1).trim());
                            }
                        }
                        readImageFromFile(headerMap, dis);
                        // Cleaning keys bound to image
                        headerMap.remove("ByteOrder");
                        headerMap.remove("Dim_1");
                        headerMap.remove("Dim_2");
                        headerMap.remove("DataType");
                        // Avoid having another item with the same name as image item
                        headerMap.remove("Image");

                        // Other DataItems
                        HashMap<String, EdfGroup> subGroupMap = new HashMap<String, EdfGroup>();
                        for (String key : headerMap.keySet()) {
                            int openIndex = key.lastIndexOf('(');
                            int closeIndex = key.lastIndexOf(')');
                            if ((openIndex > -1) && (closeIndex > openIndex)) {
                                // Group items by name
                                String groupName = key.substring(openIndex + 1, closeIndex);
                                EdfGroup subGroup = subGroupMap.get(groupName);
                                if (subGroup == null) {
                                    subGroup = new EdfGroup(null);
                                    subGroup.setName(groupName);
                                    subGroup.setShortName(groupName);
                                    subGroup.setAnalyzed(true);
                                    addSubgroup(subGroup);
                                    subGroupMap.put(groupName, subGroup);

                                }
                                String itemName = key.substring(0, openIndex).replaceAll("=", "")
                                        .trim();
                                subGroup.addDataItem(buildDataItem(itemName, headerMap.get(key)));
                            }
                            else {
                                // Build simple dataItem
                                addDataItem(buildDataItem(key, headerMap.get(key)));
                            }
                        }
                        subGroupMap.clear();

                        headerMap.clear();
                        dis.close();

                    }
                }
                catch (FileNotFoundException e) {
                    // Just ignore this case
                }
                catch (IOException e) {
                    // Just ignore this case
                }
            }
        }

        analyzed = true;
    }

    private void readImageFromFile(HashMap<String, String> headerMap, DataInputStream dis) {
        // Image Recovery
        try {
            boolean littleEndian = "LowByteFirst".equals(headerMap.get("ByteOrder"));
            int dimX = Integer.valueOf(headerMap.get("Dim_1"));
            int dimY = Integer.valueOf(headerMap.get("Dim_2"));
            String dataType = headerMap.get("DataType");
            int y = 0, x = 0;
            Object imageValue = null;
            boolean unsigned = dataType.startsWith("Unsigned");
            if ("SignedByte".equals(dataType)) {
                imageValue = new byte[dimY][dimX];
                byte[][] arrayImageValue = (byte[][]) imageValue;
                while (y < dimY) {
                    int read = dis.read();
                    if (read == -1) {
                        imageValue = null;
                        break;
                    }
                    else {

                        arrayImageValue[y][x] = (byte) read;
                        x++;
                        if (x >= dimX) {
                            x = 0;
                            y++;
                        }
                    }
                }
            }
            else if ("UnsignedByte".equals(dataType)) {

                imageValue = new short[dimY][dimX];
                short[][] arrayImageValue = (short[][]) imageValue;

                while (y < dimY) {
                    int read = dis.read();
                    if (read == -1) {
                        imageValue = null;
                        break;
                    }
                    else {
                        arrayImageValue[y][x] = (short) read;
                        x++;
                        if (x >= dimX) {
                            x = 0;
                            y++;
                        }
                    }
                }
            }
            else if ("UnsignedShort".equals(dataType) || "SignedInteger".equals(dataType)) {

                // Specific
                imageValue = new int[dimY][dimX];

                // short = 2 bytes
                // int = 4 bytes
                int factor = (unsigned) ? 2 : 4;
                int sizeToRead = factor * dimY * dimX;

                // Global
                Object flatImageValue = null;
                ByteBuffer byteBuffer = EdfFileReader.readAsBytes(sizeToRead, littleEndian, dis);

                // Specific
                if ("UnsignedShort".equals(dataType)) {
                    flatImageValue = new short[dimY * dimX];
                    ShortBuffer shortBuffer = byteBuffer.asShortBuffer();
                    shortBuffer.get((short[]) flatImageValue);
                }
                else {
                    flatImageValue = new int[dimY * dimX];
                    IntBuffer integerBuffer = byteBuffer.asIntBuffer();
                    integerBuffer.get((int[]) flatImageValue);
                }

                int globalIndex = 0;
                for (int yIndex = 0; yIndex < dimY; yIndex++) {
                    for (int xIndex = 0; xIndex < dimX; xIndex++) {
                        int value = Array.getInt(flatImageValue, globalIndex);
                        value = (unsigned) ? value & 0xffff : value;
                        ((int[][]) imageValue)[yIndex][xIndex] = value;
                        globalIndex++;
                    }
                }

            }
            else if ("SignedLong".equals(dataType) || "UnsignedInteger".equals(dataType)) {

                // Specific
                imageValue = new long[dimY][dimX];

                // long = 8 bytes
                // int = 4 bytes
                int factor = (unsigned) ? 4 : 8;
                int sizeToRead = factor * dimY * dimX;

                // Global
                Object flatImageValue = null;
                ByteBuffer byteBuffer = EdfFileReader.readAsBytes(sizeToRead, littleEndian, dis);

                // Specific
                if ("UnsignedInteger".equals(dataType)) {
                    flatImageValue = new int[dimY * dimX];
                    IntBuffer integerBuffer = byteBuffer.asIntBuffer();
                    integerBuffer.get((int[]) flatImageValue);
                }
                else {
                    flatImageValue = new long[dimY * dimX];
                    LongBuffer longBuffer = byteBuffer.asLongBuffer();
                    longBuffer.get((long[]) flatImageValue);
                }

                int globalIndex = 0;
                for (int yIndex = 0; yIndex < dimY; yIndex++) {
                    for (int xIndex = 0; xIndex < dimX; xIndex++) {
                        long value = Array.getLong(flatImageValue, globalIndex);
                        value = (unsigned) ? value & 0xFFFFFFFFL : value;
                        ((long[][]) imageValue)[yIndex][xIndex] = value;
                        globalIndex++;
                    }
                }
            }
            else if ("Signed64".equals(dataType)) {
                throw new NotImplementedException();
            }
            else if ("Unsigned64".equals(dataType)) {
                unsigned = true;
                throw new NotImplementedException();
            }
            else if ("FloatValue".equals(dataType)) {
                throw new NotImplementedException();
            }
            else if ("DoubleValue".equals(dataType)) {
                throw new NotImplementedException();
            }
            if (imageValue != null) {
                EdfDataItem imageDataItem = new EdfDataItem("Image", new DefaultArrayMatrix(
                        EdfFactory.NAME, imageValue), unsigned);
                addDataItem(imageDataItem);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
            // ignore exceptions for Image Recovery
        }
    }



    public void setAnalyzed(boolean newValue) {
        this.analyzed = newValue;
    }

    @Override
    public void addDataItem(IDataItem v) {
        if ((v instanceof EdfDataItem) && (!itemList.contains(v))) {
            itemList.add((EdfDataItem) v);
            objectList.add((EdfDataItem) v);
            v.setParent(this);
        }
    }

    @Override
    public void addOneDimension(IDimension dimension) {
        throw new NotImplementedException();
    }

    @Override
    public void addSubgroup(IGroup group) {
        if ((group instanceof EdfGroup) && (!groupList.contains(group))) {
            groupList.add((EdfGroup) group);
            objectList.add((EdfGroup) group);
            group.setParent(this);
        }
    }

    @Override
    public IDataItem findDataItem(IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem findDataItem(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem findDataItemWithAttribute(IKey key, String name, String attribute) {
        throw new NotImplementedException();
    }

    @Override
    public IDictionary findDictionary() {
        throw new NotImplementedException();
    }

    @Override
    public IGroup findGroup(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public IGroup findGroup(IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public IGroup findGroupWithAttribute(IKey key, String name, String value) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem getDataItem(String shortName) {
        if (shortName != null) {
            for (IDataItem item : itemList) {
                if (shortName.equals(item.getName())) {
                    return item;
                }
            }
        }
        return null;
    }

    @Override
    public synchronized List<IDataItem> getDataItemList() {
        try {
            if ((!isFakeGroup()) && (!isAcquisitionGroup()) && (itemList.isEmpty())) {
                analyzeEdfFile();
            }
        }
        catch (Exception e) {
            itemList.clear();
        }
        ArrayList<IDataItem> itemList = new ArrayList<IDataItem>();
        itemList.addAll(this.itemList);
        return itemList;
    }

    @Override
    public IDataItem getDataItemWithAttribute(String name, String value) {
        if (name != null) {
            for (IDataItem item : itemList) {
                IAttribute attribute = item.getAttribute(name);
                if (attribute != null) {
                    if (StringUtils.isSameString(value, attribute.getStringValue())) {
                        return item;
                    }
                }
            }
        }
        return null;
    }

    @Override
    public IDataset getDataset() {
        if (dataset == null) {
            IGroup root = getRootGroup();
            if (root == null) {
                return null;
            }
            else {
                return root.getDataset();
            }
        }
        else {
            return dataset;
        }
    }

    @Override
    public IDimension getDimension(String name) {
        throw new NotImplementedException();
    }

    @Override
    public List<IDimension> getDimensionList() {
        return new ArrayList<IDimension>();
    }

    @Override
    public IGroup getGroup(String shortName) {
        if (shortName != null) {
            for (IGroup group : groupList) {
                if (shortName.equals(group.getName())) {
                    return group;
                }
            }
        }
        return null;
    }

    @Override
    public synchronized List<IGroup> getGroupList() {
        if (groupList.isEmpty() && (!isFakeGroup())) {
            if (isAcquisitionGroup()) {
                // recover file groups
                File[] files = FileSystemView.getFileSystemView().getFiles(referenceFile, false);
                if (files != null) {
                    ArrayList<File> fileList = new ArrayList<File>();
                    fileList.addAll(Arrays.asList(files));
                    Collections.sort(fileList, new FileComparator());
                    for (File file : fileList) {
                        String path = file.getAbsolutePath();
                        int pointIndex = path.lastIndexOf('.');
                        if (file.isDirectory()
                                || ((pointIndex > -1) && "edf".equalsIgnoreCase(path
                                        .substring(pointIndex + 1)))) {
                            EdfGroup group = new EdfGroup(file);
                            group.setParent(this);
                            groupList.add(group);
                        }
                    }
                    fileList.clear();
                }
            }
            // else if (itemList.isEmpty()) {
            //
            // }
        }
        ArrayList<IGroup> result = new ArrayList<IGroup>();
        result.addAll(groupList);
        return result;
    }

    @Override
    public IGroup getGroupWithAttribute(String attributeName, String value) {
        if (name != null) {
            for (IGroup group : groupList) {
                IAttribute attribute = group.getAttribute(name);
                if (attribute != null) {
                    if (StringUtils.isSameString(value, attribute.getStringValue())) {
                        return group;
                    }
                }
            }
        }
        return null;
    }

    @Override
    public Map<String, String> harvestMetadata(String mdStandard) throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public boolean isEntry() {
        throw new NotImplementedException();
    }

    @Override
    public boolean isRoot() {
        return (dataset != null);
    }

    /**
     * Returns whether this {@link EdfGroup} is a fake group. A fake group is a group bound to no
     * {@link File} and no {@link IDataset}. These groups are used to group {@link IDataItem}s with
     * similar names
     * 
     * @return a boolean value
     */
    protected boolean isFakeGroup() {
        return ((dataset == null) && (referenceFile == null));
    }

    /**
     * Returns whether this {@link EdfGroup} is an acquisition group. An acquisition group is a
     * group bound to a directory.
     * 
     * @return a boolean value
     */
    protected boolean isAcquisitionGroup() {
        if (referenceFile == null) {
            return false;
        }
        else {
            try {
                return referenceFile.isDirectory();
            }
            catch (Exception e) {
                return false;
            }
        }
    }

    @Override
    public boolean removeDataItem(IDataItem item) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeDataItem(String varName) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeDimension(String dimName) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeDimension(IDimension dimension) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeGroup(IGroup group) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeGroup(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public void setDictionary(IDictionary dictionary) {
        throw new NotImplementedException();
    }

    @Override
    public void updateDataItem(String key, IDataItem dataItem) throws SignalNotAvailableException {
        throw new NotImplementedException();
    }

    @Override
    public void addStringAttribute(String name, String value) {
        throw new NotImplementedException();
    }

    @Override
    public String getLocation() {
        throw new NotImplementedException();
    }

    @Override
    public String getShortName() {
        return getName();
    }

    @Override
    public void setShortName(String name) {
        setName(name);
    }

    /**
     * Builds an {@link EdfDataItem} with a particular name and value.
     * 
     * @param name {@link EdfDataItem} name.
     * @param stringValue {@link EdfDataItem} string value. If this value represents a number (with
     *            or without unit), then the {@link EdfDataItem} real value is the {@link Double}
     *            representation of this value (and an attribute "unit" is added when unit is
     *            defined)
     * @return The expected {@link EdfDataItem}
     */
    private EdfDataItem buildDataItem(String name, String stringValue) {
        EdfDataItem result = null;
        // Recover Double representation and unit
        Number[] converted = convertStringToDouble(stringValue);

        EdfAttribute unit = null;
        if (converted == null) {
            char[] arrayValue = new char[stringValue.length()];
            arrayValue = stringValue.toCharArray();
            // The value is a String
            try {
                result = new EdfDataItem(name, new DefaultArrayMatrix(EdfFactory.NAME, arrayValue));
                // basicItem = new EdfDataItem(name, new BasicArray(converted, new int[] { 1 }));
            }
            catch (InvalidArrayTypeException e) {
                e.printStackTrace();
            }

        }
        else {
            // The value is a numeric value
            Number[] arrayValue = new Number[1];
            arrayValue[0] = converted[0];
            try {
                // Construction of "unit" attribute
                String tempUnit = stringValue.substring(converted[1].intValue()).trim();
                if (!tempUnit.isEmpty()) {
                    unit = new EdfAttribute("unit",
                            stringValue.substring((converted[1].intValue())));
                }
                result = new EdfDataItem(name, new DefaultArrayMatrix(EdfFactory.NAME, arrayValue));

                if (unit != null) {
                    result.addOneAttribute(unit);
                }
            }
            catch (InvalidArrayTypeException e) {
                e.printStackTrace();
            }

        }

        return result;
    }

    /**
     * This method tries to convert a {@link String} to a {@link Double}. It is more powerful than
     * {@link Double#valueOf(String)}, because it is compatible with {@link String} values that
     * contain units at the end.
     * 
     * @param toConvert The {@link String} to convert to {@link Double}
     * @return A {@link Number} array of length 2. The 1st element is the expected Double, and the
     *         2nd one is the index at which you can find the unit. Returns <code>null</code> if the
     *         conversion failed
     */
    private Number[] convertStringToDouble(String toConvert) {
        if (toConvert == null) {
            return null;
        }
        else {
            Number[] result = new Number[2];
            // Recover number part
            int index = -1;
            for (int i = toConvert.length(); i > 0; i--) {
                if (Character.isDigit(toConvert.charAt(i - 1))) {
                    index = i;
                    break;
                }
            }
            if (index > 0) {
                // A number was successfully found
                try {
                    result[0] = Double.valueOf(toConvert.substring(0, index));
                    result[1] = index;
                }
                catch (Exception e) {
                    result = null;
                }
            }
            else {
                result = null;
            }
            return result;
        }
    }

    @Override
    public String toString() {
        StringBuffer buffer = new StringBuffer();
        if (getParentGroup() != null) {
            buffer.append(getParentGroup().toString()).append("/");
        }
        buffer.append(getName());
        return buffer.toString();
    }

    @Override
    public IContainer getContainer(String shortName) {
        if (shortName != null && shortName.equals("")) {
            return this;
        }

        IGroup resultGroupItem = getGroup(shortName);
        if (resultGroupItem != null) {
            return resultGroupItem;
        }
        IDataItem resultVariableItem = getDataItem(shortName);
        if (resultVariableItem != null) {
            return resultVariableItem;
        }

        return null;
    }

    @Override
    @Deprecated
    public IContainer findContainer(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public IContainer findContainerByPath(String path) throws NoResultException {

        String[] sNodes = DefaultPath.splitStringPath(path);
        IContainer node = getRootGroup();

        // Try to open each node
        for (String shortName : sNodes) {
            if (!shortName.isEmpty() && node != null && node instanceof IGroup) {
                node = ((IGroup) node).getContainer(shortName);
            }
        }

        return node;
    }

    @Override
    public List<IContainer> findAllContainerByPath(String path) throws NoResultException {

        if (!analyzed) {
            analyzeEdfFile();
        }
        List<IContainer> list = new ArrayList<IContainer>();
        IGroup root = getRootGroup();

        // Try to list all nodes matching the path
        // Transform path into a NexusNode array
        INode[] nodes = DefaultPath.splitStringToNode(path);

        // Call recursive method
        int level = 0;
        if (PATH_SEPARATOR.equals(path)) {
            list.add(root);
        }
        else {
            list = findAllContainer(root, nodes, level);
        }

        return list;

    }

    private List<IContainer> findAllContainer(IContainer container, INode[] nodes, int level) {
        List<IContainer> result = new ArrayList<IContainer>();
        if (container != null) {
            if (container instanceof EdfGroup) {
                EdfGroup group = (EdfGroup) container;
                if (nodes.length > level) {
                    // List current node children
                    if (group.isAcquisitionGroup()) {
                        for (EdfGroup realGroup : groupList) {
                            result.addAll(findAllContainer(realGroup, nodes, level));
                        }
                    }

                    List<INode> childs = group.getNodes();

                    INode current = nodes[level];

                    for (INode node : childs) {
                        if (current.matchesPartNode(node)) {

                            if (level < nodes.length - 1) {
                                result.addAll(findAllContainer(group.getContainer(node.getName()),
                                        nodes, level + 1));
                            }
                            // Create IContainer and add it to result list
                            else {
                                result.add(group.getContainer(node.getName()));
                            }
                        }
                    }
                }
            }
            else {
                EdfDataItem dataItem = (EdfDataItem) container;
                result.add(dataItem);
            }
        }
        return result;
    }

    private List<INode> getNodes() {

        List<INode> nodes = new ArrayList<INode>();

        // if (objectList.isEmpty() && groupList.isEmpty()) {
        // analyzeEdfFile();
        // }

        for (AbstractObject object : objectList) {
            nodes.add(new DefaultNode(object.getName()));
        }

        if (isAcquisitionGroup()) {
            for (EdfGroup group : groupList) {
                nodes.add(new DefaultNode(group.getShortName()));
            }
        }

        return nodes;
    }

    @Override
    @Deprecated
    public List<IContainer> findAllContainers(IKey key) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public List<IContainer> findAllOccurrences(IKey key) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    public IContainer findObjectByPath(Path path) {
        throw new NotImplementedException();
    }

    @Override
    public long getLastModificationDate() {
        long result = 0;
        if (referenceFile != null) {
            result = referenceFile.lastModified();
        }
        return result;
    }

    @Override
    public String getFactoryName() {
        throw new NotImplementedException();
    }

    public static Object convertArrayDimensionFrom1ToN(Object singleDimArray,
            int... multiDimArrayShape) {
        Object result = null;
        if ((singleDimArray != null) && (multiDimArrayShape != null)
                && (multiDimArrayShape.length > 0) && singleDimArray.getClass().isArray()
                && (!singleDimArray.getClass().getComponentType().isArray())) {
            if (multiDimArrayShape.length == 1) {
                result = singleDimArray;
            }
            else {
                Class<?> dataType = recoverDataType(singleDimArray);
                if (Array.newInstance(dataType, 0).getClass().equals(singleDimArray.getClass())) {
                    int arrayLength = Array.getLength(singleDimArray);
                    int expectedLength = 1;
                    for (int currentLength : multiDimArrayShape) {
                        expectedLength *= currentLength;
                    }
                    if (arrayLength == expectedLength) {
                        try {
                            result = Array.newInstance(dataType, multiDimArrayShape);
                        }
                        catch (Exception e) {
                            result = null;
                        }
                        if (result != null) {
                            // java initializes int values with 0 by default
                            int[] startPositions = new int[multiDimArrayShape.length];
                            reshapeArray(0, startPositions, multiDimArrayShape, singleDimArray,
                                    result, true);
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Recursive method reshaping a filled mono-dimensional array into a multi-dimensional array, or
     * a filled multi-dimensional array into a mono-dimensional array.
     * 
     * @param dimIndex reshape all dimensions greater than this one
     * @param startPositions starting position in multi-dimensional array
     * @param dimensions dimensions' size of the multi-dimensional array
     * @param singleDimArray The mono-dimensional array
     * @param multiDimArray The multi-dimensional array
     * @param from1ToN A boolean to know the type of reshaping. If <code>TRUE</code>, then this
     *            method will reshape a filled mono-dimensional array into a multi-dimensional
     *            array. Otherwise, this method will reshape a filled multi-dimensional array into a
     *            mono-dimensional array.
     */
    private static void reshapeArray(int dimIndex, int[] startPositions, int[] dimensions,
            Object singleDimArray, Object multiDimArray, boolean from1ToN) {
        int lStartRaw;
        int lLinearStart;
        if (dimIndex == dimensions.length - 1) {
            lLinearStart = 0;
            for (int k = 1; k < dimensions.length; k++) {
                lStartRaw = 1;
                for (int j = k; j < dimensions.length; j++) {
                    lStartRaw *= dimensions[j];
                }
                lStartRaw *= startPositions[k - 1];
                lLinearStart += lStartRaw;
            }
            if ((singleDimArray != null) && (multiDimArray != null)) {
                if (from1ToN) {
                    System.arraycopy(singleDimArray, lLinearStart, multiDimArray, 0,
                            dimensions[dimensions.length - 1]);
                }
                else {
                    System.arraycopy(multiDimArray, 0, singleDimArray, lLinearStart,
                            dimensions[dimensions.length - 1]);
                }
            }
        }
        else {
            Object[] outputAsArray = (Object[]) multiDimArray;
            for (int i = 0; i < dimensions[dimIndex]; i++) {
                Object o = outputAsArray[i];
                reshapeArray(dimIndex + 1, startPositions, dimensions, singleDimArray, o, from1ToN);
                if (startPositions[dimIndex] < dimensions[dimIndex] - 1) {
                    startPositions[dimIndex] = i + 1;
                }
                else {
                    startPositions[dimIndex] = 0;
                }
            }
        }
    }

    /**
     * This methods recovers the type of data present in a N dimension array.
     * 
     * @param array The array
     * @return The {@link Class} that represents the data type in the given array. (Example: if
     *         <code>array</code> is a <code>boolean[][]</code>, the result will be
     *         {@link Boolean#TYPE}). This method returns <code>null</code> if <code>array</code> is
     *         <code>null</code>.
     */
    public static Class<?> recoverDataType(Object array) {
        Class<?> result = null;
        if (array != null) {
            result = recoverDeepComponentType(array.getClass());
        }
        return result;
    }

    /**
     * This methods recovers the type of data present in a {@link Class} that represents N dimension
     * arrays.
     * 
     * @param arrayClass The {@link Class}
     * @return The {@link Class} that represents the data type in the given array. (Example: if
     *         <code>arrayClass</code> is <code>boolean[][]</code>, the result will be
     *         {@link Boolean#TYPE}). This method returns <code>null</code> if
     *         <code>arrayClass</code> is <code>null</code>.
     */
    public static Class<?> recoverDeepComponentType(Class<?> arrayClass) {
        Class<?> result = arrayClass;
        if (arrayClass != null) {
            while (result.isArray()) {
                result = result.getComponentType();
            }
        }
        return result;
    }

    public static void main(String[] args) {
        EdfGroup group = new EdfGroup(new File(
                "/home/viguier/NeXusFiles/EDF/test_264/test_264_test_im_00.edf"));
        group.analyzeEdfFile();

    }

}
