"""
Created on Thur Feb 09 11:23 2023

@author: Jin Xianlin (xianlincn@pku.edu.cn)
@version: 1.0
@brief: An abstract class(method) describing dictionaries,
        a collection of single neuron (basis function).
@modifications: to be added
"""

from abc import ABC, abstractmethod

##=============================================##
#            an abstract basic class            #
##=============================================##

class AbstractDictionary(ABC):
    """
    @abstractmethod: abstract method, class with this decorator cannot 
                     be instantiated. A subclass who inherits the class
                     must rewite all functions with this decorator.
    """
    
    @abstractmethod
    def _select_initial_elements(self, pde_energy):
        pass
    
    @abstractmethod
    def _argmax_optimize(self, pde_energy, opt_type):
        pass
    
    @abstractmethod
    def find_optimal_element(self):
        pass
    