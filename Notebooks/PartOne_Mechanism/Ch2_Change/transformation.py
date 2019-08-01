# transformation.py
# Andrew Ribeiro
# July 2019
from abc import ABC, abstractmethod
import numpy as np
from condense_transformation import compose_fn_ls
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def repeat(elm, n):
    out = []
    for i in range(n):
        out.append(elm)
    return out


def index_dict(ls):
    dict_out = {}
    for i in range(len(ls)):
        dict_out[ls[i]] = i
    return dict_out


def flatten_list(ls):
    out_ls = []
    for elm in ls:
        if isinstance(elm, list):
            for i in flatten_list(elm):
                out_ls.append(i)
        else:
            out_ls.append(elm)
    return out_ls


class Transformation(ABC):
    @abstractmethod
    def operator(self, operand):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    def power(self, n):
        if n == 1:
            return self
        else:
            return self * self.power(n - 1)

    def power_chain(self, initial_value, n, cntr=1):
        if cntr >= n:
            return [initial_value]
        else:
            return [initial_value,
                    *self.power_chain(self.operator(initial_value), n, cntr + 1)]


class FiniteTransformation(Transformation):
    def __init__(self, domain, operator):
        self.domain = sorted(set(domain))
        self._operator = operator
        self.range = sorted(set(flatten_list(list(map(self.operator, self.domain)))))

    def operator(self, operand):
        # Returns a transform.
        if operand not in self.domain:
            raise Exception("Domain Error!")

        return self._operator(operand)

    def transitions(self):
        out = []
        for i in sorted(list(self.domain)):
            out.append((i, self.operator(i)))
        return out

    def transitions_flat(self):
        out = []
        for i in sorted(list(self.domain)):
            transform = self.operator(i)
            if not isinstance(transform, list):
                transform = [transform]
            for t in transform:
                out.append((i, t))

        return out

    def closed(self):
        for elm in self.domain:
            transform = self.operator(elm)
            if not isinstance(transform, list):
                transform = [transform]

            for t in transform:
                if not (t in self.domain):
                    return False
        return True

    def is_one_one(self):
        transitions = self.transitions()
        transforms = set()
        for transition in transitions:
            if transition[1] in transforms:
                return False
            else:
                transforms.add(transition[1])
        return True

    def matrix_representation(self):
        columns = []
        range_len = len(self.range)
        range_index_dict = index_dict(list(self.range))

        for (operand, transform) in self.transitions():
            column = np.zeros(range_len)
            for i in range(range_len):
                if not isinstance(transform, list):
                    transform = [transform]

                for j in transform:
                    column[range_index_dict[j]] = 1

            columns.append(column)

        return pd.DataFrame(np.array(columns).T, index=list(self.range), columns=list(self.domain))

    def is_one_one(self):
        if len(self.range) == 0:
            return True
        else:
            for index, row in self.matrix_representation().iterrows():
                if not sum(row) <= 1:
                    return False
            return True

    def is_single_valued(self):
        if len(self.range) == 0:
            return True
        else:
            for index, row in self.matrix_representation().T.iterrows():
                if not sum(row) <= 1:
                    return False
            return True

    def kinematic_graph(self):
        g = nx.DiGraph()

        if self.is_single_valued():
            g.add_edges_from(self.transitions())
        else:
            g.add_edges_from(self.transitions_flat())

        pos = nx.spring_layout(g, k=2.5, iterations=300)
        plt.figure(3, figsize=(12, 6))
        nx.draw_networkx(g, pos, node_color="white", arrows=True)
        nx.draw_networkx_labels(g, pos)
        plt.show()

    def __mul__(self, other):
        return FiniteTransformation(other.domain, compose_fn_ls([other._operator, self._operator]))

    @staticmethod
    def matrix_to_transformation(mat):
        nRows = mat.shape[0]
        nCols = mat.shape[1]
        operandMap = {}

        for row_i in range(nRows):
            for column_i in range(nCols):
                if mat[row_i, column_i] == 1:
                    if column_i in operandMap:
                        operandMap[column_i].append(row_i)
                    else:
                        operandMap[column_i] = [row_i]

        return FiniteTransformation(list(operandMap.keys()), lambda operand: operandMap[operand])

    @staticmethod
    def transitions_to_transformation(transitions):
        transform_dict = {}
        for (operand, transform) in transitions:
            transform_dict[operand] = transform

        return FiniteTransformation(list(transform_dict.keys()), lambda operand: transform_dict[operand])

    @staticmethod
    def list_to_transformation(ls):
        transitions = []
        for i in range(len(ls) - 1):
            transitions.append((ls[i], ls[i + 1]))
        return FiniteTransformation.transitions_to_transformation(transitions)


class VectorTransformation(Transformation):
    def __init__(self, component_functions, is_operand=lambda x: True):
        if isinstance(component_functions, list):
            def tmp_fn(vector):
                out_vector = []
                for fn in component_functions:
                    out_vector.append(fn(vector))
                return out_vector

            self.n_components = len(component_functions)
            self._operator = tmp_fn
        else:
            self._operator = component_functions

        self.is_operand = is_operand

    def operator(self, operand):
        if isinstance(operand, list) and len(operand) == self.n_components and self.is_operand(operand):
            return self._operator(operand)
        else:
            raise Exception("Domain Error!")

    def __pow__(self, n):
        return self.power(n)

    def __mul__(self, other):
        tmp = VectorTransformation(self.is_operand, compose_fn_ls([other._operator, self._operator]))
        tmp.n_components = self.n_components
        return tmp

    def __call__(self, operand):
        return self.operator(operand)


class InfiniteTransformation(Transformation):
    def __init__(self, domain_function, operator):
        self.domain_function = domain_function
        self._operator = operator

    def __mul__(self, other):
        return InfiniteTransformation(self.domain_function, compose_fn_ls([other._operator, self._operator]))

    def operator(self, operand):
        # Returns a transform.
        return self._operator(operand)

    def transitions(self, n):
        out = []
        for i in range(n):
            d = self.domain_function(i)
            out.append((d, self._operator(d)))
        return out

    def kinematic_graph(self, n):
        g = nx.DiGraph()
        g.add_edges_from(self.transitions(n))
        pos = nx.spring_layout(g, k=2.5, iterations=300)
        plt.figure(3, figsize=(12, 6))
        nx.draw_networkx(g, pos, node_color="white", arrows=True)
        nx.draw_networkx_labels(g, pos)
        plt.show()



