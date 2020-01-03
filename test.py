# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:27:06 2019

@author: lizhenping
"""

import sys 
import inspect 
  
def my_name(): 
 print ('1' ,sys._getframe().f_code.co_name )
 print ('2' ,inspect.stack()[0][3] )
  
def get_current_function_name(): 
 print ('5', sys._getframe().f_code.co_name )
 return inspect.stack()[1][3] 
class MyClass: 
 def function_one(self): 
  print( '3',inspect.stack()[0][3] )
  print( '4', sys._getframe().f_code.co_name )
  print ("6 %s.%s invoked"%(self.__class__.__name__, get_current_function_name()) )
  
if __name__ == '__main__': 
 my_name() 
 myclass = MyClass() 
 myclass.function_one()
 
import numpy as np
d = np.arange(0,2)
test = int(d)