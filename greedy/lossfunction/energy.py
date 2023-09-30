"""
Created on Mon May 15 17:34 2023

@author: Jin Xianlin (xianlincn@pku.edu.cn)
@version: 1.0
@brief: An abstract class(method) describing PDE energy functionals
        and defining standard methods.
@note: Class that inherits AbstractEnergy should be initialized with source_data,
        i.e., self.source_data = ...
@modifications: to be added
"""

from abc import ABC, abstractmethod

##=============================================##
#            an abstract basic class            #
##=============================================##


class AbstractEnergy(ABC):

    @abstractmethod
    def _get_energy_items(self, obj_func):
        pass
    
    @abstractmethod
    def _energy_norm(self, obj_func):
        pass
    
    @abstractmethod
    def energy_error(self):
        pass
    
    @abstractmethod
    def get_stiffmat_and_rhs(self, parameters, core_matrix):
        pass
    
    @abstractmethod
    def evaluate(self, param): #, pre_solution): 
        """ mark: set pre_solution as an intilaization of energy,
                  save the complexity when calling this function.
        """
        pass
    
    @abstractmethod
    def evaluate_large_scale(self, param): #, pre_solution):
        """ 
        mark_1: set pre_solution as an intilaization of energy,
                save the complexity when calling this function.
        mark_2: if base_dim=1, just call evaluate(theta). 
                Otherwise this function needs to be defined.
        """
        pass