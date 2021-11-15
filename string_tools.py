#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
from difflib import SequenceMatcher
import operator
import time
import ast
import os
import re
import numpy as np
from tqdm.std import tqdm
from urllib.parse import quote, unquote

from datetime import datetime


def get_current_time():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    return dt_string


def url2str(url):
    """解析url中最后一部分的中文字符串

    Args:
        url (str): 类似这种"http://zhishi.me/baidubaike/resource/%E6%8A%98%E5%8F%A0%E8%87%AA%E8%A1%8C%E8%BD%A6"
        将后半部分解析为中文返回

    Returns:
        str: %E6%8A%98%E5%8F%A0%E8%87%AA%E8%A1%8C%E8%BD%A6 对应的中文词汇
    """
    zh_url = unquote(url, encoding='utf-8', errors='replace')
    zh = zh_url.split(os.sep)[-1]
    return zh


def str2url(string):
    """string转为url

    Args:

    Returns:
        str:
    """
    url_suffix = quote(string, encoding='utf-8', errors='replace')
    return url_suffix


def find_lcsubstr(s1, s2):
    m = [[0 for i in range(len(s2) + 1)]
         for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长子串及其长


def lcs(X, Y, m, n):
    """得到最长公共子序列

    Args:
        X (string): 句子1
        Y (string): 句子2
        m (int): 句子1的长度
        n (int): 句子2的长度

    Returns:
        [string]: 最长公共子序列
    """
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    index = L[m][n]
    lcs = [""] * (index + 1)
    lcs[index] = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(lcs)


def lcs_split(X, Y, m, n):
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]
    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    # Following code is used to print LCS
    index = L[m][n]
    # Create a character array to store the lcs string
    lcs = [""] * (index + 1)
    lcs[index] = ""

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    continue_st = True
    while i > 0 and j > 0:
        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i - 1] == Y[j - 1]:
            split_code = ""
            if not continue_st:
                split_code = "..."
            lcs[index - 1] = X[i - 1] + str(split_code)
            i -= 1
            j -= 1
            index -= 1
            continue_st = True
        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
            continue_st = False
        else:
            j -= 1
            continue_st = False
    return "".join(lcs)


def get_lcs(st1, st2, type="subsequence"):
    """得到句子1和句子2的最长公共子序列

    Args:
        st1 (string): 句子1
        st2 (string): 句子2

    Returns:
        [string]: [公共子序列]
    """
    if type == "subsequence":
        return lcs(st1, st2, len(st1), len(st2))
    else:
        return lcs_split(st1, st2, len(st1), len(st2))


def get_lcs_in_list(lcs_list, type="subsequence"):
    common_part = lcs_list[0]
    for i, ele in enumerate(lcs_list[1:]):
        if type == "subsequence":
            common_part = get_lcs(ele, common_part, type=type)
        elif type == "subsequence_split":
            common_part = get_lcs(ele, common_part, type=type)
        elif type == "substring":
            common_part = find_lcsubstr(ele, common_part)
        print(common_part)
    return common_part


def get_punctuation():
    """返回所有标点符号

    Returns:
        [string]: 标点符号集合
    """
    punc = "\"'！!？?｡.。＂`＃#＄$％%＆&＇()（）:：＊*＋+-－／/；;＜<＝=>＞@＠[［＼］]＾^_＿｀{}｛｜｝~～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    return punc


def get_number_zh():
    number = "几一二三四五六七八九十百千万"
    return number


def get_TIME_zh():
    return "年月日时分秒"


def have_common(st1, st2):
    """判断句子1和句子2是否有公共部分

    Args:
        st1 (string): [description]
        st2 (string): [description]

    Returns:
        [Boolean]: 有则返回True
    """
    return len(get_lcs(st1, st2)) != 0


# 全角转半角
def full_to_half(text):  # 输入为一个句子
    change_text = ""
    for word in text:
        inside_code = ord(word)
        # print("before:", inside_code)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        # elif inside_code in [ord("“"), ord("”")]:
        #     inside_code = ord('"')
        # elif inside_code == ord("。"):
        #     inside_code = ord(".")
        # elif inside_code in [ord("‘"), ord("’")]:
        #     inside_code = ord("'")
        elif inside_code >= 65281 and inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        change_text += chr(inside_code)
        # print("after:", inside_code)
    return change_text


def half_to_full(text):
    """半角转全角"""
    change_text = ""
    for word in text:
        inside_code = ord(word)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        change_text += chr(inside_code)
    return change_text


def get_neighborword(idx, windows_size, token_text=None, printStep=False):
    """得到所有的邻近词

    Args:
        idx (int): 中心词下标
        windows_size (int): 邻近窗口大小
        token_text (string list): 句子分词后得到的token list

    Returns:
        [list]: 邻近词list
    """
    # if printStep:
    #     print("\t", idx, windows_size, token_text)
    neighbors = []
    length_text = len(token_text)
    # token_id_s = inputs["input_ids"][0]
    first_idx = max(idx - windows_size // 2, 0)
    neighbor_idx = [
        i for i in range(first_idx, min(first_idx +
                                        windows_size, length_text), 1)
    ]

    for ne_i in neighbor_idx:
        cur_token = token_text[ne_i]
        # cur_token = tokenizer.convert_ids_to_tokens(
        #     [token_id_s[ne_i]], skip_special_tokens=True)
        if ne_i != idx and cur_token != []:
            neighbors += cur_token
        elif ne_i == idx:
            neighbors += ["None"]
        else:
            neighbors += ["NULL"]
    return neighbors


def find_all(a_str, sub):
    """给出子串，在母串中找出所有出现下标

    Args:
        a_str (string): 母串
        sub (string): 子串

    Yields:
        start: 每次调用返回一个下标
    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_substring_indicies(text, substring="[UNK]"):
    """得到子串的下标，未分词的句子按字符得到的下标

    Args:
        text (string): 包含UNK的string

    Returns:
        [list]: 所有的UNK下标
    """
    return list(find_all(text, substring))  # [0, 5, 10, 15]


def get_all_source_ofunk(unk_text, text, tokenizer):
    """得到UNK在原文中对应的词

    Args:
        unk_text (string): 含UNK的句子
        text (string): 原句

    Returns:
        [unks]: UNK对应的原词列表
    """
    text = text.lower()
    unks = []
    for _ in get_substring_indicies(unk_text):
        # unk_text 在循环中不断更新，所以每次都需要重新划分
        unk_text_token = tokenizer.tokenize(unk_text)
        unkidx = unk_text_token.index("[UNK]")
        prefix = "".join(unk_text_token[:unkidx])
        sep = unk_text_token[unkidx + 1:unkidx + 2]
        prefix = prefix.replace("#", "")
        unk_text = unk_text.replace("#", "")
        unk = text.replace(prefix, "", 1).split("".join(sep))[0]
        unk_text = unk_text.replace(prefix + "[UNK]", "", 1)
        text = text.replace(prefix + unk, "", 1)
        unks.append(unk)
    return unks


def UNK_alignment(unk_text, unks, tokenizer):
    """将unk_text中的unk替换为原文中的原词

    Args:
        unk_text (string): 加词后的句子
        unks (string): UNK对应的原词列表

    Returns:
        deunk_text: 替换掉UNK的，加词后的句子
    """
    unk_text_token = tokenizer.tokenize(unk_text)
    added_nums = 0
    for idx, unk_token in enumerate(unk_text_token):
        if unk_token == "[UNK]":
            try:
                unk_text_token[idx] = unks[added_nums]
                added_nums += 1
            except:
                print("UNK_alignment ERROR", unk_text, unks)
    deunk_text = unk_text_token
    return deunk_text


def extract_zh(text):
    """提取中文字符

    Args:
        text ([type]): [description]
    """
    pre = re.compile(u'[\u4e00-\u9fa5]')
    res = re.findall(pre, text)
    res1 = ''.join(res)
    return res1


def detect_repeat(text, expression=r'([\u4e00-\u9fa5])(\1+)'):
    pattern = re.compile(expression)
    matchs = re.finditer(pattern, text)
    results = []
    for match in matchs:
        results.append(match.group())
    return results


def extact_content_in_bracket(text, expression=r'(\[|（).+?(）|\])'):
    pattern = re.compile(expression)
    matchs = re.finditer(pattern, text)
    results = []
    for match in matchs:
        results.append(match.group())
    return results if len(results) != 0 else None


def split_st(text, pattern=r';|。|；|,|，'):
    txts = re.split(pattern, text)
    return txts


def extract_punctuation(text):
    pattern = re.compile(r'\W')
    results = re.findall(pattern, text)
    return results


def longest_common_subsequence(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]


def red_color(text):
    return u'\033[1;31;1m%s\033[0m' % text


def green_color(text):
    return u'\033[1;32;1m%s\033[0m' % text


def find_lcsubstr(s1, s2):
    m = [
        [0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)
    ]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] >= mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1

    return s1[p - mmax: p], p - mmax, p  # 返回最长子串及相对s1的位置


def find_str(sentence, regular, group=2):
    """给定正则表达式和文本,找出符合规则的部分

    Args:
        sentence (string): 正文
        regular (string): 正则表达式
        group (int, optional): [description]. Defaults to 2.

    Returns:
        string: 目标文本
    """
    try:
        items = re.finditer(regular, sentence)
        some_str = ""
        for item in items:
            some_str += "@" + item.group(group)
    except:
        some_str = ""

    return some_str


def get_regular_expressions():
    regular_exps = []
    regular_exps.append(re.compile(r"([\"\'‘“](.+)[’”\'\"])"))  # 提出引号内容
    regular_exps.append(re.compile(r"([《<\[(](.+)[》>\])])"))  # 提出括号内容
    return regular_exps


# Returns length of LCS for X[0..m-1], Y[0..n-1]


def get_lcs(result1, ele):
    """按相似度对两个句子中被提取出来的元素进行匹配

    Args:
        result1 ([type]): [description]
        ele ([type]): [description]

    Returns:
        [type]: [description]
    """
    max_length = -1
    matched_lcs = ""
    for ori_ele in result1:
        lc_subsequence = lcs(ori_ele, ele)
        if len(lc_subsequence) > max_length:
            max_length = len(lc_subsequence)
            matched_lcs = ori_ele
    return matched_lcs


def text_alignment(text1, text2):
    """this function align the content which surround by bracket or
    quotation marks. Prevent some proper nouns changed by the reducer system.

    Args:
        text1 (string): original text
        text2 (string): text through after the paper reducer system

    Returns:
        [string]: alignment text 
    """
    regular_exps = get_regular_expressions()
    for regular_exp in regular_exps:
        result1 = find_str(text1, regular_exp).split("@")
        result2 = find_str(text2, regular_exp).split("@")
        if len(result1) != len(result2):
            return text2
        for idx, ele in enumerate(result2):
            try:
                origin_ele = result1[idx]  # 按序匹配
            except:
                print("text_alignment ERROR", text1, text2, result1, result2)
            # origin_ele = get_lcs(result1, ele)  # 按相似度匹配
            text2 = text2.replace(ele, origin_ele)
    return text2


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def compare(source, target):
    _, mapping = longest_common_subsequence(source, target)
    source_idxs = set([i for i, j in mapping])
    target_idxs = set([j for i, j in mapping])
    colored_source, colored_target = u'', u''
    for i, j in enumerate(source):
        if i in source_idxs:
            colored_source += green_color(j)
        else:
            colored_source += red_color(j)
    for i, j in enumerate(target):
        if i in target_idxs:
            colored_target += green_color(j)
        else:
            colored_target += red_color(j)
    print(colored_source)
    print(colored_target)


def extract_en(text):
    """提取英文字符

    """
    pattern = re.compile(r'[a-zA-Z]')
    results = re.findall(pattern, text)
    en = ''.join(results)
    return en


def type_restore(value):
    """将字符串恢复为其原来的数据类型

    Args:
        value (任意类型): 任意类型但被转化为了字符串

    Returns:
        数据原始类型: 将字符串格式的值恢复为其原始类型
    """
    try:
        value = ast.literal_eval(value)
    except ValueError:
        value = value
    return value


def does_b_in_a(a="hello, python", b="llo", mode=""):
    t1 = time.time()
    for i in range(5000000):
        # if b in a:
        #     pass
        # if a.find(b) == -1:
        # pass
        if operator.contains(a, b):
            pass
    print("llo" in "hello, python")
    t2 = time.time()
    print("时间:{0:0.2f}秒".format((t2 - t1)))


def colourful_text(text, color):
    """add color to text

    Args:
        text (str): [description]
        color (str): red, green, yello, blue, black
    """
    colourful = {"red": u"\033[1;31;1m%s\033[0m",
                 "green": u"\033[1;32;1m%s\033[0m",
                 "yello": u"\033[1;33;1m%s\033[0m",
                 "blue": u"\033[1;34;1m%s\033[0m",
                 "black": u"\033[1;30;1m%s\033[0m"}
    return colourful[color] % text


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


if __name__ == "__main__":
    text1 = "天天向上，好好学习"
    print(detect_repeat(text1))
