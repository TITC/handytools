import time
from sanic.response import json as sanic_json
from sanic import Sanic
from ddparser import DDParser
from LAC import LAC
from tqdm import tqdm

from cutword.ddparser.tools.struct_info.extract import CoarseGrainedInfo, FineGrainedInfo


class BaiDuTools(object):
    def __init__(self, mode="lac", use_cuda=False, use_pos=True, prob=True, * args):
        """初始化LAC和DDParser

        Args:
            mode (str, optional): The mode should be in "lac", "seg" or "rank"'. Defaults to "lac".
            use_cuda (bool, optional): 是否使用GPU. Defaults to True.
            use_pos (bool, optional): 是否使用词性标注. Defaults to True.
            prob (bool, optional): 是否返回弧的概率. Defaults to True.
        """
        super(BaiDuTools, self).__init__(*args)
        self.lac = LAC(mode=mode, use_cuda=use_cuda)
        self.ddp = DDParser(prob=prob, use_pos=use_pos, use_cuda=use_cuda)

    def debug_mode_word_pos(self, text):
        """将一句话输出为word+pos的形式

        Args:
            text ([type]): [description]
        """
        results = lac.run(text)
        for i, token in enumerate(results[0]):
            print(token + results[1][i], end="")

    def debug_colorful_ddpseg(self, text):
        """通过ddparser彩色显示各种成分

        Args:
            text ([type]): [description]
        """
        for text in [text]:
            ressult = self.ddp.parse(text)[0]
            for i, word in enumerate(ressult["word"]):
                text = (str(i + 1) + ressult["postag"][i] + word + "" +
                        str(ressult["head"][i]) + ressult["deprel"][i] + " ")
                if ressult["deprel"][i] in ("SBV", "HED", "VOB"):  # 红色
                    print(u"\033[1;31;1m%s\033[0m" % text, end="")  # 主谓宾
                elif ressult["deprel"][i] in ("ATT"):  # 绿色
                    print(u"\033[1;32;1m%s\033[0m" % text, end="")  # 定
                elif ressult["deprel"][i] in ("ADV", "F"):  # 土黄
                    print(u"\033[1;33;1m%s\033[0m" % text, end="")  # 状
                elif ressult["deprel"][i] in ("CMP"):  # 天蓝
                    print(u"\033[1;34;1m%s\033[0m" % text, end="")  # 补
                elif ressult["deprel"][i] in ("COO"):  # 天蓝
                    print(u"\033[1;34;1m%s\033[0m" % text, end="")  # 补
                else:
                    print(u"\033[1;30;1m%s\033[0m" % text, end="")
            print("\n" * 2)

    def get_words_by_pos(self, lac_output, label):
        """根据词性和专名类别标签得到对应的词列表

        Args:
            lac_output ([type]): [description]
            label ([type]): [description]

        Returns:
            [type]: [description]
        Exampl:
            name_idx = [['百度', '是', '一家', '高科技','google', '公司'],
                ['ORG', 'v', 'm', 'n','ORG', 'n']]
            ntype = "ORG"
            print(get_words_by_label(name_idx,ntype))
        """
        name_lst = lac_output[0]
        idx_lst = lac_output[1]
        idxs = [i for i, j in enumerate(idx_lst) if j == label]
        names = [name_lst[i] for i in idxs]
        return names

    def ddparser_seg(self, result):
        """对ddparser分词后的结果，根据依赖关系以及词性词性组合

        Args:
            text ([type]): [description]

        Returns:
            [type]: [description]
        """

        extend_last_word = False  # 是否右连续
        word_seg = []  # 粗粒度分词
        pos_seg = []  # 粗粒度词性
        ddptag_seg = []  # 粗粒度成分
        words = result["word"]
        postags = result["postag"]
        deptags = result["deprel"]
        heads = result["head"]  # head 列表中的值，是每个词的ID。
        ids = [i+1 for i in range(len(words))]  # 词的ID从左至右，以1计数。
        for i, word in enumerate(words):
            last_pos = postags[i - 1] if i > 0 else None
            cur_pos = postags[i]
            next_pos = postags[i+1] if i+1 < len(postags) else None
            last_head_depending = heads[i - 1] if i > 0 else None
            cur_head_depending = heads[i]
            before_last_id = ids[i - 2] if i > 0 else None
            last_id = ids[i - 1] if i > 0 else None
            cur_id = ids[i]
            next_id = ids[i+1] if i+1 < len(postags) else None
            cur_deptag = deptags[i]
            if word == "了":
                print(word)
            if deptags[i] in (
                    "ATT",  # 定中关系
                    "ADV",  # 状中关系
                    "SBV",  # 主谓关系
                    "POB",  # 介宾关系
                    "COO",  # 并列关系
                    "VOB",  # 动宾关系
                    "MT",  # 虚词成分
                    "VOB",  # 动宾关系
                    "IC",  # 子句结构
                    "CMP",  # 动补关系
                    "HED",  # 核心关系
                    "CMP"  # 动词补语
            ) and cur_pos not in ("w", "f", "u"):
                if len(word_seg) != 0 and (  # 确保prefix存在，也就是第一步不走这个分支
                        extend_last_word and
                    ((last_head_depending is cur_id  # 前一个词的head 是当前词
                     # 当前词的head 是上一个词/当前词的head是上上个词
                      or cur_head_depending is last_id or cur_head_depending is before_last_id
                      or last_head_depending is next_id  # 前一个词的head 是下一个词
                      or (last_pos, cur_pos) in (("v", "p"), ())
                      ) or len(word_seg[-1])+len(word) == 2 and last_pos != "w")):  # 如果连续两个字长度均为一且不是符号，即落单，那大概率是一个词
                    word_seg[-1] += "_"+word
                    pos_seg[-1] += "_"+cur_pos
                    ddptag_seg[-1] += "_"+cur_deptag
                else:
                    word_seg.append(word)
                    pos_seg.append(cur_pos)
                    ddptag_seg.append(cur_deptag)
                # 当前词是否允许和下一个词连续
                if (cur_deptag in ("ATT", "ADV", "VOB") and
                        cur_pos not in ("w", "f",  "u", "p")) or (cur_pos, next_pos) in (("v", "p"),):
                    extend_last_word = True
                else:
                    extend_last_word = False
            else:
                word_seg.append(word)
                pos_seg.append(cur_pos)
                ddptag_seg.append(cur_deptag)
                extend_last_word = False

        return word_seg, pos_seg, ddptag_seg

    def ddp_seg(self, text):
        """根据依赖关系+成分 分词

        Args:
            text (str): 输入文本

        Returns:
            str: 分词后结果
        """
        text_result = self.ddp.parse(text)[0]
        return self.ddparser_seg(text_result)

    def ddp_seg_batch(self, texts):
        """调研ddparser进行批量自定义分词

        Args:
            texts (str): 输入文本

        Returns:
            list: 分词结果
        """
        results = []
        texts_seg = self.ddp.parse(texts)
        for ele in tqdm(texts_seg, "seging..."):
            results.append(self.ddparser_seg(ele))
        return results

    def ddparser_whole_proofread(self, text1, text2):
        texts_list = []  #
        head_list = []
        deprel_list = []
        for text in [text1, text2]:
            ressult = self.ddp.parse(text)[0]
            texts_list.append(ressult["word"])
            head_list.append(ressult["head"])
            deprel_list.append(ressult["deprel"])

        # 获取插入词下标
        for i, ele in enumerate(texts_list[1]):
            if ele != texts_list[0][i]:
                # print(ele)
                break
        # print(i)
        # 从加词后的句子中删除该词
        # print("加词组 删词前：\n", texts_list[1], len(texts_list[1]))
        # print("非加词组 ：\n", texts_list[0], len(texts_list[0]))
        del texts_list[1][i]
        # 同时 删除head中对应的下标
        # print("加词组 删词后：\n", texts_list[1], len(texts_list[1]))
        # print("加词组 下标更新前：\n", head_list[1])
        del head_list[1][i]

        # 同时删除对应的依赖关系deprel
        del deprel_list[1][i]
        # print("加词组 下标数目更新后\n", head_list[1])
        # print("原词下标：\n", head_list[0], "数目：\n", len(head_list[0]))
        # 由于加词后占有1个位置，更新加词组下标
        head_list[1] = [ele - 1 if ele >= i else ele for ele in head_list[1]]
        # print("加词后 下标值更新后\n", head_list[1], "数目\n", len(head_list[1]))
        l1 = head_list[0]
        l2 = head_list[1]

        def compare2list(l1, l2):
            same = True
            for i, ele in enumerate(l2):
                if ele != l1[i]:
                    # print(ele, l1[i])
                    same = False
                    break
            return same

        return (compare2list(l1, l2), l1, l2, i, deprel_list[0], deprel_list[1])

    def ddparser_neighbor_proofread(text1, text2):
        """输入加如1个词前后的句子，返回加词前后的前一个和后一个词的依赖序号，返回插入词在两句话中位置的前后词的依赖关系

        Args:
            text1 ([type]): [description]
            text2 ([type]): [description]

        Returns:
            [type]: [description]
        """
        _, l1, l2, i, dep1, dep2 = self.ddparser_whole_proofread(text1, text2)
        # print(l1[i], l2[i])  # 加入位置后一个词
        # print(l1[i - 1], l2[i - 1])  # 加入位置前一个词
        # print(dep1[i], dep2[i])
        # print(dep1[i - 1], dep2[i - 1])
        return (
            (l1[i], l2[i]),  # 插入词后的head
            (l1[i - 1], l2[i - 1]),  # 插入词前的head
            (dep1[i], dep2[i]),  # 插入词后的依赖关系
            (dep1[i - 1], dep2[i - 1]),  # 插入词前的依赖关系
        )

    def get_lac_pos(self):
        """得到lac词性列表
        """
        text = """n	普通名词	f	方位名词	s	处所名词	nw	作品名
        nz	其他专名	v	普通动词	vd	动副词	vn	名动词
        a	形容词	ad	副形词	an	名形词	d	副词
        m	数量词	q	量词	r	代词	p	介词
        c	连词	u	助词	xc	其他虚词	w	标点符号
        PER	人名	LOC	地名	ORG	机构名	TIME	时间"""
        pos = []
        for t in text.split("\n"):
            for ele in t.split("\t"):
                if (ele.strip().encode("utf-8").isalpha()):
                    pos.append(ele.strip())
        return pos

    def ddp_grained_seg(self, texts, grain="Coarse"):
        """按照粒度分词

        Args:
            texts (list): 输入文本list
            grain (str, optional): 分词粒度，候选项为"Coarse","Fine". Defaults to "Coarse".

        Returns:
            [type]: [description]
        """
        grained_ddp = {"Coarse": CoarseGrainedInfo,  # 粗粒度
                       "Fine": FineGrainedInfo}  # 细粒度
        infos = []
        if isinstance(texts, list):
            for text in texts:
                ddp_res = self.ddp.parse(text)
                info = grained_ddp[grain](ddp_res[0])
                infos.append(info.parse())
        return infos


# %%
if __name__ == "__main__":
    baidutools = BaiDuTools()
    texts = """RNN是一种包含一个或多个反馈环的神经网络。由于反馈在神经网络中的应用形式不同，导致rnn的结构也不同。在学习算法中，梯度下降法得到了广泛的应用。1989年，Williams和Zipser提出了实时循环学习算法(RTRL)[21]。文献[22]给出了单神经元RNN的确定收敛性梯度下降算法。本文对rnn的梯度学习算法进行了理论研究。
    """.split("\n")
    # baidutools.debug_colorful_ddpseg(text)
    print(baidutools.ddp_seg_batch(texts))
