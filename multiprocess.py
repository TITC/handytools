import math
from multiprocessing import Pool
import numpy as np
import pickle

from tqdm.std import tqdm


class MultiProcess(object):
    def __init__(self, *args):
        super(MultiProcess, self).__init__(*args)

    def execute(self, function, batch_datas: list, i: int, params: dict):
        results = []
        for text in tqdm(batch_datas, desc="%d is processing..." % i):
            result = function(text, params, i)
            results.append(result)
        return results

    def start(self, function, data, process_num=10):
        """开启多进程

        Args:
            function (python函数): 实际处理单个数据的函数。有3个形参function(data, params,i):
            data (dict): 数据. data = {"datas":[],"params":{}}
            process_num (int, optional): 默认开启的进程数目. Defaults to 10.
        """
        pool = Pool(process_num)
        output_pool, output = [], []
        datas = data["datas"]
        params = data["params"]
        batch_size = math.ceil(len(datas)/process_num)
        for i in range(process_num):
            batch_datas = datas[i*batch_size:(i+1)*batch_size]
            output_pool.append(pool.apply_async(
                func=self.execute, args=(function, batch_datas, i, params)))

        pool.close()
        pool.join()

        for i in output_pool:
            output.extend(i.get())
        self.output = output


if __name__ == '__main__':
    def get_num(data, params):
        return data[-1]

    multiprocessfun = MultiProcess()
    data = ["今天是星期%d" % (i+1) for i in range(7)]*100
    multiprocessfun.start(function=get_num, data={
        "datas": data, "params": ""}, process_num=10)
    print(multiprocessfun.output)
