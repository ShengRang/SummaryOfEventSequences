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
            pe = min(ne/self.n, 1-1e-6)
            t = (i, ne, pe)
            self.model.append(t)
        #print 'get model: '
        #pprint(self.model)
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
        #print 'bound: '
        #print self.bound
        for i, v in enumerate(self.bound):
            if v == 0 or i == 0:
                continue
            a = self.prev_bound(i)
            if a is None:
                continue
            px = self.bit_tree.query(a, i-1) / ((i - a)*self.n)
            """
            print 'a: %d, i: %d' % (a, i)
            print 'sn: %d' % (self.bit_tree.query(a, i-1), )
            print 'px : %f' % (px, )
            """
            f = lambda x: px**x[1] * (1-px)**(self.n-x[1])
            for i in range(a, i):
                res *= f(self.model[i])
        return res

    def prev_bound(self, i):
        """
        :param i:
        :return: i的前一个边界, 复杂度O(n), 可以考虑采用别的方式进行优化(例如保存起来并维护)
        """
        a = max(x[0] for x in enumerate(self.bound) if x[1] and x[0] < i)
        return a

    def next_bound(self, i):
        """
        :param i: 边界位置
        :return: 返回i的后一个边界
        """
        b = min(x[0] for x in enumerate(self.bound) if x[1] and x[0] > i)
        return b

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
        self.bound = array([1]*(self.n+1))

    def input(self):
        for i in range(self.m):
            self.S[i] = array([int(x) for x in raw_input().split()])

    def show_s(self):
        print self.S

    def find(self):
        """
        贪心或者dp去搞把..
        """
        raise NotImplementedError()


class GreedySegmentUnit(BaseSegmentUnit):
    def __init__(self, *args, **kwargs):
        super(GreedySegmentUnit, self).__init__(*args, **kwargs)
        self.delta = np.zeros(self.m+1, dtype=np.float64)

    def init_delta(self):
        """
        初始化所有的delte. (全部有边界)
        """
        m = self.m
        delta = self.delta
        delta[0] = delta[m] = 123   #两端无法去除
        self.bound[:] = 1
        for i, v in enumerate(self.bound):
            if i == 0 or i == m:
                continue
            self.update_delta(i)

    def update_delta(self, pos):
        """
        :param pos: 位置
        :return: 更新pos处的delta值
        """
        n = self.n
        m = self.m
        if pos == 0 or pos == m:
            return
        bit_tree = self.bit_tree
        delta = self.delta
        prev_bound, next_bound = self.prev_bound, self.next_bound
        a, b = prev_bound(pos), next_bound(pos)
        delta[pos] = -2 * log(m, 2)    # delta(lm)
        p_ai = min(bit_tree.query(a, pos-1) / ((pos - a) * n), 1.0 - 1e-6)
        p_ib = min(bit_tree.query(pos, b-1) / ((b - pos) * n), 1.0 - 1e-6)
        p_ab = min(bit_tree.query(a, b-1) / ((b - a) * n), 1.0 - 1e-6)
        #print p_ai, p_ib, p_ab
        log_p_ai, log_p_ib, log_p_ab = map(lambda x: log(x, 2), (p_ai, p_ib, p_ab))
        log_1p_ai, log_1p_ib, log_1p_ab = map(lambda x: log(x, 2), (1-p_ai, 1-p_ib, 1-p_ab))
        """
        计算delta(ld)
        """
        for k in range(a, pos):
            nk = bit_tree.query(k, k)
            delta[pos] += nk*(log_p_ai-log_p_ab) + (n-nk)*(log_1p_ai-log_1p_ab)
        for k in range(pos, b):
            nk = bit_tree.query(k, k)
            delta[pos] += nk*(log_p_ib-log_p_ab) + (n-nk)*(log_1p_ib-log_1p_ab)

    def find(self):
        self.init_delta()
        delta = self.delta
        bound = self.bound
        prev_bound, next_bound = self.prev_bound, self.next_bound
        update_delta = self.update_delta
        while True:
            idx, val = np.argmin(delta), min(delta)
            print '本轮查询最小值及下标: (val: %.3f, idx: %d)' % (val, idx)
            if val >= 0:
                break
            else:
                bound[idx] = 0
                delta[idx] = 123    #移除边界
                a, b = prev_bound(idx), next_bound(idx)
                update_delta(a)
                update_delta(b)
        print '完成段内分组: '
        print bound
        return self.ll


class GreedyEventSeq(BaseEventSeq):
    def __init__(self, *args, **kwargs):
        super(GreedyEventSeq, self).__init__(*args, **kwargs)
        self.delta = np.zeros(self.m+1, dtype=np.float64)

    def init_delta(self):
        pass

    def find(self):
        self.init_delta()


if __name__ == '__main__':
    n, m = [int(x) for x in raw_input().split()]
    xx = BaseEventSeq(n, m)
    xx.input()
    xx.show_s()
    y = GreedySegmentUnit(xx.S[:, 0:12])
    #y.show()

    y.bound[1] = 0
    print 'pr: ' + str(y.pr)
    print 'ld: ' + str(y.ld)
    print 'lm: ' + str(y.lm)
    print 'll: ' + str(y.ll)

    print y.find()

    xx.show_s()
