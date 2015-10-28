#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import numpy as np
from numpy import array
from math import log


class BaseSegmentUnit(object):
    def __init__(self, arr):
        self.M = arr
        self.m, self.n = arr.shape
        self.bound = array([1]*self.m)

    def show_s(self):
        print self.M

    @property
    def l(self):
        return sum(1 for x in self.bound if x == 1)

    @property
    def lm(self):
        return 2*self.l*log(self.m, 2) + self.m*log(self.m, 2)

    @property
    def ld(self):
        return -log(self.pr, 2)

    @property
    def ll(self):
        return self.ld+self.lm

    @property
    def pr(self):
        pass

    def prev_bound(self, i):
        """
        :param i:
        :return: i的前一个边界, 复杂度O(n), 可以考虑采用别的方式进行优化(例如保存起来并维护)
        """
        a = min(x[0] for x in enumerate(self.bound) if x[1])
        return a if a < i else None

    def next_bound(self, i):
        b = max(x[0] for x in enumerate(self.bound) if x[1])
        return b if b > i else None


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


if __name__ == '__main__':
    xx = BaseEventSeq(3, 4)
    xx.input()
    xx.show_s()
    y = BaseSegmentUnit(xx.S[:, 1:3])
    y.show_s()
