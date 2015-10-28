# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


BIT_TREE_DEFAULT_SIZE = 500
lowbit = lambda x: x & -x


class BitTree(object):
    """
    在维护区间和的时候可以利用树状数组优化
    为了支持数组性质, 这里的树状数组在更新, 查询前对x进行-1的偏置(支持x==0)
    self.bit 为树状数组,
    self.raw 为原始数组
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        单例模式树状数组, 节省开销.
        """
        if cls._instance is None:
            cls._instance = super(BitTree, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, maxsize=BIT_TREE_DEFAULT_SIZE):
        """
        初始化树状数组的maxsize.
        """
        self.maxsize = maxsize
        if not hasattr(self, 'bit') or len(self.bit) < maxsize+1:
            self.bit = np.zeros(maxsize+1, dtype=int)
        if not hasattr(self, 'raw') or len(self.raw) < maxsize+1:
            self.raw = np.zeros(maxsize+1, dtype=int)
        self.bit[:] = 0
        self.raw[:] = 0

    def update(self, x, val):
        """
        x处单点更新val的增量
        :param x: 更新位置, 注意0<=x<maxsize
        :param val: 增量
        :return: None
        """
        x += 1
        self.raw[x] += val
        while x < self.maxsize:
            self.bit[x] += val
            x += lowbit(x)

    def sum(self, x):
        """
        查询前x项和
        """
        x += 1
        res = 0
        while x > 0:
            res += self.bit[x]
            x -= lowbit(x)
        return res

    def query(self, l, r):
        """
        区间查询和
        """
        if l <= 0:
            return sum(r)
        if l == r:
            return self.raw[r]
        return sum(r) - sum(l-1)


if __name__ == '__main__':
    """
        just test it
    """
    bt = BitTree(50)
    bt.update(1, 5)
    print bt.sum(1)
    print bt.sum(3)
    bt.update(3, 14)
    print bt.sum(1)
    print bt.sum(2)
    print bt.sum(3)