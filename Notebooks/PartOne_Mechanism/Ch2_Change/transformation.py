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


class FiniteTransformation:
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

    def power(self, n):
        if n == 1:
            return self
        else:
            return self * self.power(n-1)

    def power_chain(self, tansformation, initial_value, n, cntr=1):
        if cntr >= n:
            return initial_value
        else:
            return [initial_value,
                    *self.power_chain(tansformation, tansformation.power(cntr).operator(initial_value), n, cntr + 1)]

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



class InfiniteTransformation(FiniteTransformation):
    def __init__(self):
        self.operandFunction = None
        self.transformFunction = None

    @abstractmethod
    def operator(self, operand):
        # Returns a transform.
        if not self.is_an_operand(operand):
            raise Exception("Domain Error!")

    def is_an_operand(self, x):
        # Checks to see if x is an operand. Assumes convexity.
        guess = 1
        reversing = False
        while True:
            result = self.operandFunction(guess)
            if x > result:
                if not reversing:
                    guess *= 2
                else:
                    # We are oscillating
                    return False
            elif x == result:
                return True
            else:
                reversing = True
                guess -= 1

    def closed(self, start=1, stop=100):
        """
        Check if this transformation is closed on a range of operands.
        """
        for i in range(start, stop):
            if not self.is_an_operand(self.transformFunction(self.operandFunction(i))):
                return False
        return True


