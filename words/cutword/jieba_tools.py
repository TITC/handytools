# encoding=utf-8
import jieba.posseg as psg
import time
import jieba
from multi_process import MultiProcess
import paddle
from tqdm.std import tqdm
from file_tools import readtexts
# paddle.enable_static()
# jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
# jieba.enable_parallel(20)


class JiebaTools(object):
    def __init__(self, dict_path: str = None, *args):
        super(JiebaTools, self).__init__(*args)
        paddle.enable_static()
        if dict_path != None:
            jieba.load_userdict(dict_path)
            jieba.initialize()  # build the prefix dictionary

    def add_word(self, word):
        jieba.add_word(word)

    def test(self):
        strs = ["我来到北京清华大学", "乒乓球拍卖完了", "中国科学技术大学"]
        for str in strs:
            seg_list = jieba.cut(str, use_paddle=True)  # 使用paddle模式
            print("Paddle Mode: " + '/'.join(list(seg_list)))

        seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
        print("Full Mode: " + "/ ".join(seg_list))  # 全模式

        seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
        print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

        seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
        print(", ".join(seg_list))

        seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
        print(", ".join(seg_list))

        seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
        print(", ".join(seg_list))

    def run(self, txts, show_bar=False):
        """输入模仿LAC的格式，分词单独在一个list，POS在另一个list。

        Args:
            txts (list): 文本列表

        Returns:
            [type]: 二维矩阵，每一行是一句话的分词。该行有两个list，前一个是分词list，后一半是POS list
        """
        result = []
        w = None
        if show_bar:
            w = tqdm(txts, desc=u'已加载0个text文件')
        else:
            w = txts
        for i, st in enumerate(w):
            cur_words = []
            cur_pos = []
            seg_list = psg.cut(st)
            for word, pos in seg_list:
                cur_words.append(word)
                cur_pos.append(pos)
            result.append([cur_words, cur_pos])
            if show_bar:
                w.set_description(u'已加载%s个text文件' % str(i+1))
        return result

    def default_tokenize(self, data, params, i):
        return list(jieba.cut(data.replace(" ", "")))

    def multiprocess_cut(self, path, process_num=10, function=None):
        if function is None:
            function = self.default_tokenize
        self.multiprocess = MultiProcess()
        lines = [line.replace('\n', '')
                 for line in open(path, "r", encoding='utf-8') if len(line.replace('\n', '')) > 1]
        self.multiprocess.start(function=function, data={
            "datas": lines, "params": ""}, process_num=process_num)
        return self.multiprocess.output


if __name__ == '__main__':
    texts = ["我来到xxxxxxxxxxxxxxx北京清华大学", "乒乓球拍卖完了", "中国科学技术大学"]
    jiebatools = JiebaTools()
    result = jiebatools.run(texts, show_bar=True)
    print(result)
    # path = "/yuhang/draft/data/test_data/fictions/wjtx/test.txt"
    # jiebatools = JiebaTools()
    # print(jiebatools.multiprocess_cut(path=path))
    # def default_tokenize(data, params, i):
    #     return list(jieba.cut(data))
    # print(default_tokenize("今天是星期二", None, None))
