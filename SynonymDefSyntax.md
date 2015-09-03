# Introduction #

The **synonym document is relative to a particular view**. Synonyms have a meaning only when using a the dictionary mechanism. It will permit to associate for each plug-in a specific keyword to another one.

# The need of synonyms #

The need of synonyms appears when an application can use several plug-ins. It will ask for data using keywords from its defined view.

Mainly, there's two way of defining keywords of a view:
  * in worst cases, according a particular data format
  * or according a concept file relative to a specific experiment

The fact, is that there is absolutely **no guarantee that all plug-ins use same keywords** or directly refers to a concept file (which should be the good way of doing). So keywords used by the application can be unidentified in the mapping of a specific plug-in. Even if the physical measure exist under an another form in the mapping file.

**This is where synonyms interpose**.

# What for #

Synonyms are loaded at the same time as the view. Its main purpose is to link two keywords. Once linked and when called both of them will return the same data.

The synonym of a keyword is plug-in dependent. It means that the specified synonym will be used only when the associated plug-in is working.

All synonyms corresponding to a view are stored in the same XML file. When specified that particular is searched in the same folder as the view.

# Use case #


Let's imagine an application that only displays images, it uses the following view:

```
<!DOCTYPE data-def SYSTEM "../dtd/view_file.dtd">
<data-def name="IMAGE_VIEWER" synonym=""> 
    <group key="data"> 
        <item key="images"/>
        <item key="x_position"/> 
        <item key="y_position"/> 
    </group> 
    <group key="info"> 
        <item key="comments"/> 
    </group> 
</data-def>
```

and two plug-ins are available, they have defined the following mapping:

  * plug-in A:
```
<!DOCTYPE map-def SYSTEM "../../dtd/mapping_file.dtd">
<map-def name="plug_in_A" version="1.0.0">
...
  <item key="images">
    <path>/entry1/data/my_images</path>
  </item>
...
</map-def>
```

  * plug-in B
```
<!DOCTYPE map-def SYSTEM "../../dtd/mapping_file.dtd">
<map-def name="plug_in_B" version="1.0.0">
...
  <item key="image_stack">
    <path>/image*/data</path>
    <call>DataItemStacker</call>
  </item>
...
</map-def>
```

The application will use **the keyword _images_ according its view**. In the displayed view file above, no synonym file is mentioned. Therefore, in the case of the _plug-in A_, the CDMA will answer an item, whereas **the _plug-in B_ won't find the item**.

Now in the view, **we define the synonym file name** as the following:
```
<data-def name="IMAGE_VIEWER" synonym="my_synonym_file.xml"> 
```

In the same folder **we create the synonym file** so called "my\_synonym\_file.xml", as following:
```
<?xml version="1.0" encoding="UTF8"?>
<!DOCTYPE synonym-def SYSTEM "../dtd/synonyms_file.dtd">
<synonym-def>
   <plugin name="plug_in_B">
      <synonym key="images" mapping="image_stack"/>
   </plugin>
</synonym-def>
```

Therefore, the application will still ask for data with both plug-ins the same way. But the in the case of the _plug-in B_ **the keyword _images_ will be translated into _image\_stack_**.