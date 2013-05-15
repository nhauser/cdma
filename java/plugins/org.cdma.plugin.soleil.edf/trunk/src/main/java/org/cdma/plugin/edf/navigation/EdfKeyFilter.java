//package org.cdma.plugin.edf.navigation;
//
//import org.cdma.dictionary.filter.IFilter;
//
//public class EdfKeyFilter implements IFilter, Cloneable {
//
//    private FilterLabel label;
//    private Object value;
//
//    public EdfKeyFilter(FilterLabel label, Object value) {
//        super();
//        setLabel(label);
//        setValue(value);
//    }
//
//    @Override
//    public IFilter clone() {
//        try {
//            return (IFilter) super.clone();
//        }
//        catch (CloneNotSupportedException e) {
//            // Should not happen because this class is Cloneable
//            return null;
//        }
//    }
//
//    @Override
//    public FilterLabel getLabel() {
//        return label;
//    }
//
//    @Override
//    public Object getValue() {
//        return value;
//    }
//
//    @Override
//    public void setLabel(FilterLabel label) {
//        this.label = label;
//    }
//
//    @Override
//    public void setValue(Object value) {
//        this.value = value;
//    }
//
//    @Override
//    public boolean equals(IKeyFilter keyfilter) {
//        return equals((Object) keyfilter);
//    }
//
//    @Override
//    public boolean equals(Object obj) {
//        if ((obj != null) && (getClass().equals(obj.getClass()))) {
//            EdfKeyFilter filter = (EdfKeyFilter) obj;
//            if (label == filter.getLabel()) {
//                if (value == null) {
//                    return (filter.getValue() == null);
//                }
//                else {
//                    return (value.equals(filter.getValue()));
//                }
//            }
//        }
//        return false;
//    }
//
//    @Override
//    public int hashCode() {
//        int code = 34;
//        code = code * 71 + (label == null ? 0 : label.hashCode());
//        code = code * 71 + (value == null ? 0 : value.hashCode());
//        return code;
//    }
//
// }
