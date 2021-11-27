import os
from file_tools import files_with_extension, readpkl, readjson, read_tsvfile, parent_dir, save2pkl
import sys
import sanic
from sanic import Sanic
from sanic.response import json as sanic_json
import ast
from tqdm.std import tqdm
from string_tools import type_restore


class WordFreq(object):
    init_flag = False
    sogou_data = None
    ztflh_data = None
    national_data = None
    repo_root = parent_dir(os.path.realpath(__file__), 3)
    freq_root = os.path.join(
        repo_root, "data", "auxiliary", "word_frequency")

    def __init__(self, *args):
        if WordFreq.init_flag:  # 防止重复加载文件
            return
        # 加载 现代汉语语料库词频表
        national_path = os.path.join(
            WordFreq.freq_root, "national", "word_frequences.json")
        self.ztflh_path = os.path.join(
            WordFreq.freq_root, "field", "txtperline_2021-10-26", "csv")
        self.mapping_path = self.ztflh_path+os.sep+"mapping.pkl"
        WordFreq.national_data = readjson(national_path)
        # 加载搜狗语料库
        WordFreq.sogou_data = self.load_sogou()
        # 加载中图分类法语料库
        # WordFreq.ztflh_data = self.load_ztflh()
        # 进行术语到属于类别的映射
        # WordFreq.w2ztfl = self.construct_word_2_ztflh_mapping()

        # 加载处理好的中图分类法语料库
        WordFreq.w2ztfl = readpkl(self.mapping_path)

        WordFreq.init_flag = True

    def load_sogou(self):
        """解析搜狗词库为key value形式

        Returns:
            dict: key是word，value是word frequency
        """
        word_freq = {}
        sogou_path = os.path.join(
            WordFreq.freq_root, "sogou", "SogouLabDic.dic")
        # Using readlines()
        file1 = open(
            sogou_path,
            "r",
        )
        Lines = file1.readlines()
        # Strips the newline character
        for line in Lines:
            cur_line = line.split("\t")
            word_freq[cur_line[0]] = cur_line[1]
        return word_freq

    def load_ztflh_by_major(self, ztflh_path: str, csv_filename: str):
        # 加载中图分类法领域词汇

        rows, column_titles = read_tsvfile(
            os.path.join(ztflh_path, csv_filename))
        column_titles[0] = "ID_count"
        column_titles.append("ID_TF-IDF")  # 追加TF-IDF排名
        major_dict = {}
        for row_idx, row in enumerate(rows):
            row.append(row_idx)  # 追加TF-IDF排名
            major_dict[row[1]] = row
        return column_titles, major_dict

    def load_ztflh(self):
        extension = "csv"
        filepaths = files_with_extension(self.ztflh_path, extension)
        column_title = None
        fields = {}
        for filename in tqdm(filepaths, desc="filed glossary is loading... "):
            column_title, major_dict = self.load_ztflh_by_major(
                self.ztflh_path, filename)
            fields[filename.replace(".csv", "")] = major_dict
        return column_title, fields

    def get_ztflh_word_frequency(self, field: str, word: str):
        """按词取csv的一行

        Args:
            word (str): [description]

        Returns:
            [type]: [description]
        """
        column_titles, fields = self.ztflh_data  # 返回csv文件列名，领域词库
        try:
            field_dict = fields.get(field)
            value = field_dict.get(word)
            result = {}
            for i, ele in enumerate(value):
                ele = type_restore(value=ele)
                column_title = column_titles[i]
                result[column_title] = ele
            return result
        except Exception:
            return None

    def construct_word_2_ztflh_mapping(self):
        """构建术语词汇到中图分类法类别的映射

        Returns:
            dict: key是word，value 是类别list
        """
        column_titles, fields = self.ztflh_data  # 返回csv文件列名，领域词库
        word_idx = column_titles.index("word")
        w2ztfl = {}
        for field in tqdm(fields, desc="constructing..."):
            for word in fields[field]:
                # info = self.get_ztflh_word_frequency(word=word, field=field)
                # 0是依据频次排名
                id_count = type_restore(value=fields[field][word][0])
                # 2是频次
                count = type_restore(value=fields[field][word][2])
                # -1是追加的TF-IDF排名
                id_tfidf = type_restore(value=fields[field][word][-1])
                info = {column_titles[0]: id_count,
                        column_titles[2]: count,
                        column_titles[-1]: id_tfidf}
                if w2ztfl.get(word) is None:
                    w2ztfl[word] = {field: info}
                else:
                    w2ztfl[word][field] = info

        save2pkl(self.mapping_path, w2ztfl)
        return w2ztfl

    def get_common_word_frequency(self, word: str, source="national"):
        """读取词频表

        Args:
            word ([type]): [description]
            source ([type]): [description]

        Returns:
            [type]: [description]
        """
        if source == "national":
            # http://corpus.zhonghuayuwen.org/resources/CorpusIntroduction2012.pdf
            return self.national_data.get(word)
        elif source == "sogou":
            frequency = self.sogou_data.get(word)
            if frequency is not None:
                frequency = int(frequency)
            return frequency


if __name__ == '__main__':
    word_freq1 = WordFreq()
    word = "计算机"
    #########################测试对常用词库的词频读取#################################################
    result = word_freq1.get_common_word_frequency(
        word=word, source="national")
    print(result)
    #########################测试对领域词库的词频读取#################################################
    result = word_freq1.get_ztflh_word_frequency(field="TP", word=word)
    print(result)
    #########################词2类#################################################
    result = word_freq1.w2ztfl[word]
    print(result)
