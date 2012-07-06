#!/usr/bin/env python

import unittest 

import dataset_test
import group_test
import dataitem_test

suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(dataset_test))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(group_test))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(dataitem_test))

runner = unittest.TextTestRunner()
result = runner.run(suite)

