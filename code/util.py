# -*- coding: utf-8 -*-

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
        maxsize = max(maxsize, BIT_TREE_DEFAULT_SIZE)
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
        if l == r:
            return self.raw[r+1]
        return self.sum(r) - self.sum(l-1)


class Heap(object):
    """
    一个堆.. 记录了进入堆的元素在堆的什么位置....
    用来优化论文算法
    """
    def __init__(self, key=lambda x: x, data=None):
        self.heap = []
        self.heap.extend(data)
        self.index = range(len(self.heap))
        self.re_index = range(len(self.heap))
        self.hlength = len(self.heap)
        self.total = len(self.heap)
        self.key = key
        """
        print 'raw heap: '
        print self.heap
        print 'raw length:'
        print self.hlength
        print 'raw index: '
        print self.index
        """

    def heapify(self):
        hlength = self.hlength
        down = self.down
        for i in range((hlength-1)/2+1)[::-1]:
            down(i)
        return

    def up(self, p):
        """
        p处元素上浮, O(log n)
        """
        heap = self.heap
        index = self.index
        re_index = self.re_index
        key = self.key
        q = (p-1) / 2
        a = heap[p]
        r = re_index[p]
        while q >= 0 and key(heap[q]) > key(a):
            heap[p] = heap[q]
            re_index[p] = re_index[q]
            index[re_index[q]] = p
            p = q
            q = (p-1) / 2
        heap[p] = a
        re_index[p] = r
        index[r] = p
        return

    def down(self, p):
        """
        p除元素下沉 O(log n)
        """
        heap = self.heap
        index = self.index
        re_index = self.re_index
        hlength = self.hlength
        key = self.key
        q = p*2 + 1
        a = heap[p]
        r = re_index[p]
        while q < hlength:
            if q+1 < hlength and key(heap[q+1]) < key(heap[q]):
                q += 1
            if key(heap[q]) >= key(a):
                break
            heap[p] = heap[q]
            re_index[p] = re_index[q]
            index[re_index[q]] = p
            p = q
            q = p*2 + 1
        heap[p] = a
        re_index[p] = r
        index[r] = p
        return

    def get_index(self, i):
        """
        返回最初第i个进入堆的元素, 在heap[]数组中当前位置
        """
        return self.index[i]

    def delete(self, p):
        """
        返回heap[pos]处的元素 O(log n)
        """
        hlength = self.hlength
        heap = self.heap
        re_index = self.re_index
        index = self.index
        if 0 <= p < hlength:
            heap[p] = heap[hlength-1]
            index[re_index[hlength-1]] = p
            re_index[p] = re_index[hlength-1]
            self.up(p)
            self.down(p)
            hlength -= 1

    def heap_pop(self):
        """
        删除堆顶元素 O(log n)
        """
        tmp = self.heap[0]
        self.delete(0)
        return tmp

    def heap_push(self, val):
        """
        插入新元素到堆中 复杂度O(log n)
        """
        heap, hlength, re_index, index, total = self.heap, self.hlength, self.re_index, self.index, self.total
        if len(heap) <= hlength:
            heap.extend([0]*(hlength-len(heap)+1))
        heap[hlength] = val
        if len(re_index) <= hlength:
            re_index.extend([0]*(hlength-len(re_index)+1))
        re_index[hlength] = total
        index.append(hlength)    #index[total] = hlength
        hlength, total = hlength+1, total+1
        self.up(hlength-1)

    def modify(self, pos, val):
        """
        将heap[pos]修改为val. 然后up, down调整堆
        复杂度O(log n)
        """
        if 0 <= pos < self.hlength:
            self.heap[pos] = val
            self.up(pos)
            self.down(pos)
        return

    @property
    def empty(self):
        return self.hlength <= 0


if __name__ == '__main__':
    """
        just test it
    """
    """
    x = BitTree()
    x.update(0, 11)
    x.update(1, 10)
    x.update(2, 2)

    print x.query(0, 0)
    print x.query(1, 1)
    print x.query(2, 2)

    print x.query(0, 1)
    print x.query(1, 2)

    print x.query(0, 2)
    """
    # test the heap
    x = [-5, -7, -9, -55]
    heap = Heap(key=lambda t: t[1], data=enumerate(x))
    heap.heapify()
    print heap.heap
    print heap.index
    print heap.re_index

    print heap.heap_pop()
    print heap.heap
