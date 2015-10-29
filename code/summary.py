#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import numpy as np
from numpy import array
from math import log
from pprint import pprint

from util import BitTree


class BaseSegmentUnit(object):
    def __init__(self, arr):
        self.M = arr
        self.m, self.n = arr.shape
        self.bound = array([1]*(self.m+1))
        self.bit_tree = BitTree(maxsize=self.m)
        #self.delta = np.zeros(self.m, dtype=float)
        self.model = []
        for i in range(self.m):
            ne = sum(1 for x in self.M[i] if x == 1)
            t = (i, ne, ne/self.n)
            self.model.append(t)
        print 'get model: '
        pprint(self.model)
        self.model = sorted(self.model, key=lambda x: -x[1])
        for i in range(len(self.model)):
            self.bit_tree.update(i, self.model[i][1])
        print 'after sort:'
        pprint(self.model)

    def show(self):
        print '获取的部分S[]数组'
        print self.M
        print 'model:'
        pprint(self.model)

    @property
    def l(self):
        """
        :return: 当前划分得到几个段内分组
        """
        return sum(1 for x in self.bound if x == 1)-1

    @property
    def lm(self):
        """
        :return: 当前划分得到的Lm
        """
        return 2*self.l*log(self.m, 2) + self.m*log(self.m, 2)

    @property
    def ld(self):
        """
        :return: 当前划分得到的Ld
        """
        return -log(self.pr, 2)

    @property
    def ll(self):
        """
        :return: Ll = Ld + Lm
        """
        return self.ld+self.lm

    @property
    def pr(self):
        """
        当前分组得到的Pr
        :return: Pr  (Ld = -log(Pr, 2))
        """
        res = 1.0
        for i, v in enumerate(self.bound):
            if v == 0:
                #没有边界则不计算
                continue
            a = self.prev_bound(i)
            if a is None:
                continue
            px = self.bit_tree.query(a, i-1) / ((i - a)*self.n)
            f = lambda x: px**x[1] * (1-px)**(self.n-x[1])
            for i in range(a, i):
                res *= f(self.model[i])
        return res

    def prev_bound(self, i):
        """
        :param i:
        :return: i的前一个边界, 复杂度O(n), 可以考虑采用别的方式进行优化(例如保存起来并维护)
        """
        a = min(x[0] for x in enumerate(self.bound) if x[1])
        return a if a < i else None

    def next_bound(self, i):
        """
        :param i: 边界位置
        :return: 返回i的后一个边界
        """
        b = max(x[0] for x in enumerate(self.bound) if x[1])
        return b if b > i else None

    def find(self):
        """
        :return: 进行算法步骤, 返回一个值表示最优的Ll, 并将结果保存进self.model, self.bound.
        可以考虑将结果用更好的方式进行保存.
        """
        raise NotImplementedError()


class BaseEventSeq(object):
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.S = array([0]*m*n).reshape(m, n)
        #print self.S
        self.bound = array([1]*self.n)

    def input(self):
        for i in range(self.m):
            self.S[i] = array([int(x) for x in raw_input().split()])

    def show_s(self):
        print self.S


class GreedySegmentUnit(BaseSegmentUnit):
    def __init__(self, *args, **kwargs):
        super(GreedySegmentUnit, self).__init__(*args, **kwargs)
        self.delta = np.zeros(self.m, dtype=float)

    def find(self):
        pass


if __name__ == '__main__':
    n, m = [int(x) for x in raw_input().split()]
    xx = BaseEventSeq(n, m)
    xx.input()
    xx.show_s()
    y = GreedySegmentUnit(xx.S[:, 0:12])
    y.show()
    y.bound[1] = 0
    print 'pr: ' + str(y.pr)
    print 'ld: ' + str(y.ld)
    print 'lm: ' + str(y.lm)
    print 'll: ' + str(y.ll)
