package org.cdma.plugin.edf.navigation;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.filechooser.FileSystemView;

import org.cdma.dictionary.Path;
import org.cdma.exception.NoResultException;
import org.cdma.exception.SignalNotAvailableException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.plugin.edf.abstraction.AbstractGroup;
import org.cdma.plugin.edf.abstraction.AbstractObject;
import org.cdma.plugin.edf.array.BasicArray;
import org.cdma.plugin.edf.utils.FileComparator;
import org.cdma.plugin.edf.utils.StringUtils;

public class EdfGroup extends AbstractGroup {

    protected static final String DEFAULT_SHORT_NAME = "data";

    private final File referenceFile;
    private final ArrayList<AbstractObject> objectList;
    private final ArrayList<EdfDataItem> itemList;
    private ArrayList<EdfGroup> groupList;
    private final EdfDataset dataset;

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
    }

    public EdfGroup(File file) {
        this(null, file);
    }

    /**
     * Analyzes the bound EDF {@link File} to create children {@link IGroup}s and {@link IDataItem}s
     */
    private void analyzeEdfFile() {
        if (referenceFile != null) {
            HashMap<String, String> headerMap = new HashMap<String, String>();
            try {
                BufferedInputStream reader = new BufferedInputStream(new FileInputStream(
                        referenceFile.getAbsolutePath()));
                int character = -1;
                int count = 0;
                StringBuffer headerBuffer = new StringBuffer();
                boolean newLine = false;
                while (true) {
                    try {
                        count++;
                        character = reader.read();
                        if (character == -1) {
                            break;
                        }
                        headerBuffer.append((char) character);
                        if (character == '\n') {
                            newLine = true;
                        }
                        else {
                            if ((character == '}') && newLine) {
                                count++;
                                character = reader.read();
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
                            headerMap.put(line.substring(0, separatorIndex).trim(), line.substring(
                                    separatorIndex + 1).trim());
                        }
                    }
                    // Image Recovery
                    try {
                        boolean littleEndian = "LowByteFirst".equals(headerMap.get("ByteOrder"));
                        int dimX = Integer.valueOf(headerMap.get("Dim_1"));
                        int dimY = Integer.valueOf(headerMap.get("Dim_2"));
                        String dataType = headerMap.get("DataType");
                        int y = 0, x = 0;
                        Number[][] imageValue = null;
                        boolean unsigned = false;
                        if ("SignedByte".equals(dataType)) {
                            imageValue = new Byte[dimY][dimX];
                            while (y < dimY) {
                                int read = reader.read();
                                if (read == -1) {
                                    imageValue = null;
                                    break;
                                }
                                else {
                                    imageValue[y][x] = (byte) read;
                                    x++;
                                    if (x >= dimX) {
                                        x = 0;
                                        y++;
                                    }
                                }
                            }
                        }
                        else if ("UnsignedByte".equals(dataType)) {
                            unsigned = true;
                            imageValue = new Short[dimY][dimX];
                            while (y < dimY) {
                                int read = reader.read();
                                if (read == -1) {
                                    imageValue = null;
                                    break;
                                }
                                else {
                                    imageValue[y][x] = (short) read;
                                    x++;
                                    if (x >= dimX) {
                                        x = 0;
                                        y++;
                                    }
                                }
                            }
                        }
                        else if ("SignedShort".equals(dataType)) {
                            imageValue = new Short[dimY][dimX];
                            byte[] bitPack = new byte[2];
                            int lastBinaryValue;
                            while (y < dimY) {
                                lastBinaryValue = reader.read(bitPack);
                                if (lastBinaryValue == bitPack.length) {
                                    if (littleEndian) {
                                        // extract corresponding unsigned short (little endian)
                                        imageValue[y][x] = (short) (((bitPack[1] << (short) 8)) | (bitPack[0]));
                                    }
                                    else {
                                        // extract corresponding unsigned short (big endian)
                                        imageValue[y][x] = (short) ((bitPack[0] << (short) 8) | (bitPack[1]));
                                    }
                                    x++;
                                    if (x >= dimX) {
                                        x = 0;
                                        y++;
                                    }
                                }
                                else {
                                    imageValue = null;
                                    break;
                                }
                            }
                        }
                        else if ("UnsignedShort".equals(dataType)) {
                            imageValue = new Integer[dimY][dimX];
                            byte[] bitPack = new byte[2];
                            int lastBinaryValue;
                            while (y < dimY) {
                                lastBinaryValue = reader.read(bitPack);
                                if (lastBinaryValue == bitPack.length) {
                                    imageValue[y][x] = 0;
                                    if (littleEndian) {
                                        // extract corresponding unsigned short (little endian)
                                        imageValue[y][x] = ((bitPack[1] & 0xff) << 8)
                                                | ((bitPack[0] & 0xff));
                                    }
                                    else {
                                        // extract corresponding unsigned short (big endian)
                                        imageValue[y][x] = ((bitPack[0] & 0xff) << 8)
                                                | ((bitPack[1] & 0xff));
                                    }
                                    x++;
                                    if (x >= dimX) {
                                        x = 0;
                                        y++;
                                    }
                                }
                                else {
                                    imageValue = null;
                                    break;
                                }
                            }
                        }
                        else if ("SignedInteger".equals(dataType) || "SignedLong".equals(dataType)) {
                            // TODO Integer
                        }
                        else if ("UnsignedInteger".equals(dataType)
                                || "UnsignedLong".equals(dataType)) {
                            unsigned = true;
                            // TODO unsigned Integer
                        }
                        else if ("Signed64".equals(dataType)) {
                            // TODO Long
                        }
                        else if ("Unsigned64".equals(dataType)) {
                            unsigned = true;
                            // TODO unsigned Long
                        }
                        else if ("FloatValue".equals(dataType)) {
                            // TODO
                        }
                        else if ("DoubleValue".equals(dataType)) {
                            // TODO
                        }
                        if (imageValue != null) {
                            EdfDataItem imageDataItem = new EdfDataItem("Image", new BasicArray(
                                    imageValue, new int[] { dimX, dimY }), unsigned);
                            addDataItem(imageDataItem);
                        }
                    }
                    catch (Exception e) {
                        // ignore exceptions for Image Recovery
                    }
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
                    reader.close();
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
        // TODO Auto-generated method stub
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
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IDataItem findDataItem(String shortName) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IDataItem findDataItemWithAttribute(IKey key, String name, String attribute) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IDictionary findDictionary() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IGroup findGroup(String shortName) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IGroup findGroup(IKey key) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IGroup findGroupWithAttribute(IKey key, String name, String value) {
        // TODO Auto-generated method stub
        return null;
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
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<IDimension> getDimensionList() {
        // TODO Auto-generated method stub
        return null;
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
            else if (itemList.isEmpty()) {
                // recover item groups
                analyzeEdfFile();
            }
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
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public boolean isEntry() {
        // TODO Auto-generated method stub
        return false;
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
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public boolean removeDataItem(String varName) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public boolean removeDimension(String dimName) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public boolean removeDimension(IDimension dimension) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public boolean removeGroup(IGroup group) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public boolean removeGroup(String shortName) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public void setDictionary(IDictionary dictionary) {
        // TODO Auto-generated method stub

    }

    @Override
    public void updateDataItem(String key, IDataItem dataItem) throws SignalNotAvailableException {
        // TODO Auto-generated method stub

    }

    @Override
    public void addStringAttribute(String name, String value) {
        // TODO Auto-generated method stub

    }

    @Override
    public String getLocation() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public String getShortName() {
        return DEFAULT_SHORT_NAME;
    }

    @Override
    public void setShortName(String name) {
        // Not managed
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
        Object value = null;
        // Recover Double representation and unit
        Number[] converted = convertStringToDouble(stringValue);
        EdfAttribute unit = null;
        if (converted == null) {
            // The value is a String
            value = stringValue;
        }
        else {
            // The value is a numeric value
            value = converted[0];
            try {
                // Construction of "unit" attribute
                String tempUnit = stringValue.substring(converted[1].intValue()).trim();
                if (!tempUnit.isEmpty()) {
                    unit = new EdfAttribute("unit", stringValue.substring(converted[1].intValue()));
                }
            }
            catch (Exception e) {
                unit = null;
            }
        }
        EdfDataItem basicItem = new EdfDataItem(name, new BasicArray(value, new int[] { 1 }));
        if (unit != null) {
            basicItem.addOneAttribute(unit);
        }
        return basicItem;
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
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    @Deprecated
    public IContainer findContainer(String shortName) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IContainer findContainerByPath(String path) throws NoResultException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<IContainer> findAllContainerByPath(String path) throws NoResultException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    @Deprecated
    public List<IContainer> findAllContainers(IKey key) throws NoResultException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    @Deprecated
    public List<IContainer> findAllOccurrences(IKey key) throws NoResultException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IContainer findObjectByPath(Path path) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public long getLastModificationDate() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public String getFactoryName() {
        // TODO Auto-generated method stub
        return null;
    }

}
