
import pympler.asizeof
from guppy import hpy
import sys
import time
from tqdm.std import tqdm
from file_tools import delete_file, readtexts, save2pkl
from string_tools import extract_zh


class Constant(object):
    error = "ERROR"
    success = "SUCCESS"
    failure = "FAILURE"

    def __init__(self, *args):
        super(Constant, self).__init__(*args)


class Node(object):
    __slots__ = ('_children', '_frequency', '_char')
    size = 0  # accumulate the number of all nodes

    def __init__(self, char=None) -> None:
        self._children = list()  # private
        self._frequency = 1  # private
        self._char = char  # private

    def _put(self, excerpt: str):
        # prevent it's None
        if excerpt is None:
            return Constant.error
        # iterate the excerpt
        child = self
        for char in excerpt:
            child = child._add_child(char)
        return Constant.success

    def construct_bintrie(self, window: int, path: str):
        texts = readtexts(path)
        for text in tqdm(texts, desc="save excerpts"):
            # if text length is less than window, then skip
            if len(text) < window:
                continue
            # if Chinese ratio is less than 0.5, then skip
            if len(extract_zh(text)) / len(text) < 0.5:
                continue
            # iterate the text
            for i in range(len(text)-window):
                excerpt_str = text[i:i+window]
                self._put(excerpt_str)

    def __contains__(self, key):
        return self[key] is not None

    def __getitem__(self, key):
        state = self
        for char in key:
            index, found = self._binary_search(state._children, char)
            if found:
                state = state._children[index]
            else:
                return None
        return state._frequency

    def _add_child(self, char):
        index, found = self._binary_search(self._children, char)
        if found:
            self._children[index]._frequency += 1
        else:
            new_node = Node(char)
            Node.size += 1
            self._children.insert(index, new_node)
        return self._children[index]

    def _binary_search(self, datas, char: str):
        found = False
        low = 0
        high = len(datas) - 1
        while low <= high:
            mid = (low + high) >> 1
            if datas[mid]._char == char:
                found = True
                return mid, found
            elif datas[mid]._char < char:
                low = mid + 1
            else:
                high = mid - 1
        return low, found


if __name__ == '__main__':
    node = Node()

    path = "xxxx.txt"

    node.construct_bintrie(5, path)
    print(len(node._children))
    print(node.size)
    print("MB:", pympler.asizeof.asizeof(node)/1024/1024)
    # h = hpy()
    # stats = h.heap()
    # print("Total Objects : ", stats.count)
    # print("Total Size : ", stats.size/1024/1024, "Miga Bytes")
    # print(stats)
    path = path.replace(".txt", ".pkl")
    delete_file(path)
    save2pkl(node, path)
    time_start = time.time()
    for i in range(5):
        if "小魔女" in node:
            print("小魔女")
        # if "是他的背影" in node:
        #     print("是他的背影")
        if "是他的背部" in node:
            print("是他的背部")
    time_end = time.time()
    # print with 3 decimal places
    print("time: {:.3f}s".format(time_end - time_start))
    # node.start = True
