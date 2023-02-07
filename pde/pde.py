"""
Created on Tue Dec 13 22:10 2022

@author: Jin Xianlin (xianlincn@pku.edu.cn)
@version: 1.0
@brief: An abstract class(method) built for PDE data
        used in training neural networks.
@modifications: to be added
"""

from abc import ABC, abstractmethod


##=============================================##
#            an abstract basic class            #
##=============================================##
class PDE(ABC):  
    """
    @abstractmethod: abstract method, class with this decorator cannot 
                     be instantiated. A subclass who inherits the class
                     must rewite all functions with this decorator.
    """
    
    @abstractmethod
    def dimension(self):
        return self.dim
        
    @abstractmethod
    def equation_order(self):
        return self.order
        
    @abstractmethod
    def solution(self, p):
        return 
    
    @abstractmethod
    def gradient(self, p):
        return 
    
    @abstractmethod
    def source(self, p):
        return
        