import networkx as nx
import numpy as np
import perceval as pcvl
from perceval.components.unitary_components import BS
from scipy.optimize import minimize
import matplotlib.pyplot as plt; plt.rcdefaults()
from perceval.algorithm import Sampler
import time
from perceval.utils import InterferometerShape
from perceval.components import catalog

def parify_samples(samples, parity):
    """apply the parity function to the samples"""    
    # ...
    return new_samples

def set_parameters_circuit(parameters_circuit, values): 
    """set values of circuit parameters"""
    for idx, p in enumerate(parameters_circuit):
        parameters_circuit[idx].set_value(values[idx])

def compute_samples(circuit, input_state, nb_samples, parity): 
    """sample from the circuit"""
    # ...
    sampler=Sampler(processor)
    samples = sampler.sample_count(nb_samples)['results']
    return parify_samples(samples, parity)

def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.

    Args:
        x: str
           solution bitstring

        G: networkx graph

    Returns:
        obj: float
             Objective
    """
    # ...
    return obj

def graph_to_Q(G: nx.Graph):
    """
    Computes expectation value based on measurement results

    Args:
        counts: dict
                key as bitstring, val as count

        G: networkx graph

    Returns:
        avg: float
             expectation value
    """


    # ...
    return Q

def run_configuration(circuit, parity, n, G, nb_samples):
    """output the samples for a given configuration of parity and n (includes minimisation of the loss function)"""
    
    nb_modes = # ...
    input_state = # ...
    parameters_circuit = circuit.get_parameters()

    def loss(parameters):
        set_parameters_circuit(parameters_circuit, parameters)
        samples_b = # ...
        E = 0; sum_count = 0
        # compute expectation value

        return # expectation value 

    init_parameters = # ...
    best_parameters = # ...
    set_parameters_circuit(parameters_circuit, best_parameters)
    samples = compute_samples(circuit, input_state, nb_samples, parity)
    return samples

def expectation_value(vec_state, matrix, offset):
    """
    Towards generic cases. Function that returns the value for qubo given the configuration
    """
    return 

############################# EVERYTHING TOGETHER #############################

def qubo_solver(H, nb_samples = 2048):
    """run the universal circuit and optimize the parameters.
        arguments:
         - G - networkX graph to compute the Max Cut
         - nb_samples
    """
    start_time = time.time()
    if type(H)!=type(nx.graph):
        nb_modes = len(H)
    else: 
        nb_modes = H.number_of_nodes()
        H = graph_to_Q(H)
    circuit = pcvl.Circuit.generic_interferometer(nb_modes, lambda i: BS(theta=pcvl.P(f"theta{i}")))
    
    # parity and ns represent the four required tests according to the QUBO with LO paper
    parity = [0, 1]
    ns = [nb_modes, nb_modes-1]
    E_max = 10e6
    for j in parity:
        for n in ns:
            current_sample = run_configuration(circuit, j, n, H, nb_samples)
            # ...

    results_dictionary = {
        "average_cut": E_max,
        "best_state": best_state_sample, #pcvl BasicState
        "best_state_res": best_state_res,
        "time": time.time()-start_time,
        "optimised_circuit_output": configuration_samples
    }
    return results_dictionary