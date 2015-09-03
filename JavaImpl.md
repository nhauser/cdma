Welcome to the Java section of the CDMA documentation. It describes how to use the CDMA library to help the development of data analysis applications.

This section is divided into six sections:


  * an [Introduction](#Introduction.md) explaining the Java version design and guidelines.

  * a section for data analysis application developers: [Client application development](JavaClient.md).

  * an introduction section for contributors describing the [programmation guidelines](JavaGuideLines.md) to follow when adding new functionalities to the core library or working on new or existing data format engines.

  * the [data format engine development](JavaEngine.md) guide

  * the [Plug-in development](JavaPlugin.md) guide

  * finally the automatically generated DOxygen documentation of the CDMA source code [API Reference guide](JavaReference.md)


# Introduction #

The Java implementation of the CDMA was initially developed above the NetCDF data format to extract a common interface layer. The idea was to add further data format according those interfaces.
The project after several enhancing iterations has now the ability:
  * to manage several different format
  * to detect which one is the most appropriate
  * to abstract its physical structure
  * to give a common layer and tools to manipulate big arrays

In 2012 it has reached a maturity that permits to different institutes (ANSTO and SOLEIL) using different data format to share a common GUI application: the DataBrowser.


This Java implementation uses several modules:

  * [core library](#Core.md)
  * [engines libraries](#Engines.md)
  * [data plug-ins](#Plug-ins.md)

In addition some [tools are provided too](JavaTools.md).

## Core ##

The core library contains:

  * the `Factory` class: entry point for client applications,
  * a set of interfaces that must be implemented by the engines (may be override by plug-ins),
  * the dictionary mechanism,
  * the plug-ins detection mechanism,
  * the plug-ins system management (loading, method invocation).
  * a set of tools for arrays manipulations


## Engines ##

Engines are packages that implement all the needed code to handle physical data formats. A same data format engine can be used by several different plug-ins (see below). They can't be directly used by the core library, but only through the plug-ins mechanism.

## Plug-ins ##

Data plug-in encapsulates all the institute specific way of organizing data into physical containers (files, database, ...). Loaded at runt time, they drive the underlying engine to get an access to the institute data source.

## Components schema ##

The schema below summarizes the dependencies between the different software components, from an application point of view.

![http://cdma.googlecode.com/svn/wiki/images/java_archi.png](http://cdma.googlecode.com/svn/wiki/images/java_archi.png)