#!/usr/bin/env python

import unittest 

import dataset_test
import group_test

suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(dataset_test))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(group_test))

runner = unittest.TextTestRunner()
result = runner.run(suite)

