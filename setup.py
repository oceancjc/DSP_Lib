# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize
import os 

files = ['dataprocessing','instruments']
for i in files:    
    setup( ext_modules = cythonize(i+'.py') )
    os.remove(i+'.c')
