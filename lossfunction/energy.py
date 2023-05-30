"""
Created on Mon May 15 17:34 2023

@author: Jin Xianlin (xianlincn@pku.edu.cn)
@version: 1.0
@brief: An abstract class(method) describing PDE energy functionals
        and defining standard methods.
@note: Class that inherits Energy should be initialized with source_data,
        i.e., self.source_data = ...
@modifications: to be added
"""

from abc import ABC, abstractmethod

##=============================================##
#            an abstract basic class            #
##=============================================##


class Energy(ABC):

    @abstractmethod
    def get_bilinear_form(self, obj_func):
        pass
    
    @abstractmethod
    def evaluate(self, theta): #, pre_solution): 
        """ mark: set pre_solution as an intilaization of energy,
                  save the complexity when calling this function.
        """
        pass
    
    @abstractmethod
    def evaluate_large_scale(self, theta): #, pre_solution):
        """ 
        mark_1: set pre_solution as an intilaization of energy,
                save the complexity when calling this function.
        mark_2: if base_dim=1, just call evaluate(theta). 
                Otherwise this function needs to be defined.
        """
        pass
    
    @abstractmethod
    def get_stiffness_matrix(self, parameters, core_matrix):
        pass