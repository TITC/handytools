# -*- coding: utf-8 -*-
from __future__ import print_function
from email.mime import text
from enum import Flag
from posixpath import dirname
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from os import path
from time import time
import os
import zipfile
from datetime import datetime
import shutil
import glob
import os
from tqdm import tqdm
from string_tools import full_to_half
import jsonlines
import json
# importing csv module
import csv
import os
import shutil
import pandas as pd
import traceback


def get_newpath(pypath, filename: str, subfolder=""):
    """传入调用路径，文件名，子文件夹名，返回新的路径
    Args:
        pypath (str): 调用路径
        filename (str): 文件名
        subfolder (str, optional): 子文件夹名. Defaults to "".
    Returns:
        str: 新的路径
    """
    pypath = parent_dir(pypath, 1)
    if subfolder:
        newpath = path.join(pypath, subfolder, filename)
    else:
        newpath = path.join(pypath, filename)
    return newpath


def read_lines_as_list(file_path):
    with open(file_path) as f:
        List = f.read().splitlines()
    return List


def union_dict(dict1, dict2):
    for key in dict1.keys():
        if dict2.get(key) != None:
            dict2[key] += dict1[key]
        else:
            dict2[key] = dict1[key]
    return dict2


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
    return None


def get_splited_dict(dictionary, split_nums):
    dict_lengths = len(dictionary)
    batch_size = dict_lengths // split_nums
    batch_dict = []
    for n in range(split_nums + 1):
        cur_idx = batch_size * n
        end_idx = batch_size * (n + 1)
        cur_batch = dict(list(dictionary.items())[cur_idx:end_idx])
        batch_dict.append(cur_batch)
    return batch_dict


def testsplit_dict():
    dictionary = {}
    split_nums = 10
    dict_size = 208
    for n in range(dict_size):
        dictionary[str(n)] = str(n + 1)

    batch_dict = get_splited_dict(dictionary, split_nums)
    for idx, bd in enumerate(batch_dict):
        print(idx, len(bd), type(bd))


def sort_dict_by_value(d, increase=True):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=not increase))


def parent_dir(path: str, layers: int = 1):
    """按指定的父级层数，取对应的目录

    Args:
        path (str): 当前文件名
        layers (int, optional): 父目录层数，当前文件的上一级目录为1，再往上则为2，以此类推. Defaults to 1.

    Returns:
        [str]: n层以上的父级目录
    """
    dirname = os.path.dirname
    for l in range(layers):
        path = dirname(path)
    return path


def glob_text(text_path, read_fun, stop_i=None,  show_bar=False):
    """对给定路径下所有符合命名规则的文本按行进行读取

    Args:
        text_path (string): 模糊匹配文件名

    Returns:
        [list]: 按行读取后的文本list
    """
    text_list = []
    pathlist = glob.glob(text_path)
    if show_bar:
        w = tqdm(pathlist, desc=u'已加载0个text')
    else:
        w = pathlist
    for i, txt in enumerate(w):
        if stop_i is not None and i > stop_i:
            break
        text_list.extend(read_fun(txt, show_bar=False))
        if show_bar:
            w.set_description(u'已加载%s个text' % str(i+1))
    return text_list, pathlist


def readtexts(path, portion=1, end: int = None, show_bar=True, encoding="utf-8"):
    """将txt的每一行读出来

    Args:
        txt_path (string): 文件路径
        portion (int, optional): 读出文件比例，比如只需要对一部分看看. Defaults to 1.
        end (int, optional): 指定读到哪一行为止. Defaults to None.
        showBar (bool, optional): 是否显示进度条. Defaults to True.
        encoding (str,optional): 'UTF-8-sig'  "utf-8" 'urt8'
    Returns:
        list: each elements is a row
    """
    data_file = []
    with open(path, "r+", encoding=encoding) as f:
        if end is None:
            num_lines = len([1 for line in open(path, "r", encoding=encoding)])
            num_lines *= portion
        else:
            num_lines = end
        if show_bar:
            for idx, item in enumerate(
                    tqdm(f, total=num_lines, desc=path.split(os.sep)[-1] + " is loading...")):
                if idx >= num_lines:
                    break
                data_file.append(item.replace("\n", ""))
        else:
            for idx, item in enumerate(f):
                if idx >= num_lines:
                    break
                data_file.append(item.replace("\n", ""))
    return data_file


def readjsonline(jsonl_path, show_bar=False):
    """read jsonline file

    Args:
        jsonl_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    with open(jsonl_path, "r+", encoding="utf8") as f:
        lines = len([1 for line in open(
            jsonl_path, "r", encoding="utf-8")])
        data_file = []
        for i, item in enumerate(tqdm(jsonlines.Reader(f), total=lines, desc=u'jsonl is loading...')):
            data_file.append(item)
    return data_file


def print_files_size_in_list(root_path, path_list):
    for path in path_list:
        _, _, MB, _ = file_size(os.path.join(root_path, path))
        print("%s size is %s MB" % (path, MB))


def head_n_pkl(path, nums):
    with open(path, "rb") as f:
        dict_object = pickle.load(f)
        i = 0
        for key, val in tqdm(dict_object.items()):
            i += 1
            if i > nums:
                break
            print(key, val)


def save2pkl(obj, save_path):
    f = open(save_path, "wb")
    pickle.dump(obj, f)
    f.close()


def readpkl(dict_path):
    with open(dict_path, "rb") as f:
        dict_object = pickle.load(f)
    return dict_object


def resavejson2txt(raw_path, tgt_path, key_name):
    num_lines = len([1 for line in open(raw_path, "r", encoding="utf-8")])

    with open(raw_path, "r+", encoding="utf8") as f:
        data_file = []
        for idx, item in enumerate(jsonlines.Reader(tqdm(f, total=num_lines))):
            tgt = item.get(key_name)
            tgt = full_to_half(tgt)
            data_file.append(tgt)

    with open(tgt_path, "w") as f:
        for item in data_file:
            f.write("%s\n" % item)


def generate_emptyfolder_bylist(root_path, folders_list):
    """generate empty folder by folders name

    Args:
        root_path (string): [description]
        folders_list (list): [description]
    """
    floder_list = []
    for fol in folders_list:
        floder_list.append(os.path.join(root_path, fol))
    cleanfolder(floder_list)


def generate_cleanfolder(folder_path):
    """clean directory by path recursively

    Args:
        folder_path (string): folder path
    """
    folder_paths = []
    if isinstance(folder_path, list):
        folder_paths.extend(folder_path)
    else:
        folder_paths.append(folder_path)
    for floder in folder_paths:
        # print('make folder:',floder)
        if not os.path.isdir(floder):
            os.mkdir(floder)
    for folder in folder_paths:
        if not os.path.isdir(folder):
            print("this folder name doesn't exist...")
            break
        files = os.listdir(folder)
        for filename in files:
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # print("shutil :",file_path)
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_all_sub_folders(path):
    sub_directory = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isdir(file_path):
            sub_directory.append(filename)
    return sub_directory


def make_dir_if_not_exist(dir_path):
    """make directory if dir_path not exist

    Args:
        dir_path (string): directory absolute path

    Returns:
        [status]: if successfully created directory return True,
                  nontheless return False
    """
    status = True
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except Exception:
        status = False
    return status


def files_with_extension(data_path, extension):
    """get files by filename extension

    Args:
        data_path ([type]): [description]
        extension ([type]): [description]

    Returns:
        file_list: file name list
    """
    file_list = [
        file for file in os.listdir(data_path)
        if file.lower().endswith(extension)
    ]
    return file_list


def file_size(path):
    """输入文件路径，依次输出bytes，KB,MB，以及GB大小

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    size = os.path.getsize(path)
    return size, round(size / (1024**2),
                       2), round(size / (1024**2),
                                 2), round(size / (1024**3), 2)


def SendMail(subject,
             ImgPath,
             extension,
             opt_dict,
             time_consume,
             email="",
             pwd="",
             files=[]):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = email
    msg['To'] = email

    text = MIMEText("time consumption:" + str(round(time_consume / 3600, 3)) +
                    "h,\n")
    msg.attach(text)


def Average(lst):
    return sum(lst) / len(lst)


def savelist(file_path, my_list):
    try:
        with open(file_path, 'w') as f:
            for item in tqdm(my_list, desc=file_path.split(os.sep)[-1] + " is saving..."):
                f.write("%s\n" % item)
        return True
    except Exception as ex:
        print(ex)
        print(file_path, np.array(my_list).shape)
        return False


def append_list2file(file_path, my_list, show_bar=False):
    try:
        with open(file_path, 'a+') as f:
            if show_bar:
                for item in tqdm(my_list, desc=file_path.split(os.sep)[-1] + " is saving..."):
                    f.write("%s\n" % item)
            else:
                for item in my_list:
                    f.write("%s\n" % item)
        return True
    except Exception as ex:
        print(ex)
        print(file_path, np.array(my_list).shape)
        return False


def readjsonl_bykey(jsonl_path, key_name, skip=-1):
    """按key读取json文件中的value

    Args:
        jsonl_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    num_lines = len([1 for line in open(jsonl_path, "r", encoding="utf-8")])
    with open(jsonl_path, "r+", encoding="utf8") as f:
        data_file = []
        for idx, item in enumerate(
                jsonlines.Reader(
                    tqdm(f, total=num_lines, desc="loading repeat source."))):
            tgt = item.get(key_name)
            data_file.append(tgt)
    return data_file


def read_cnsd2txt(txt_path):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    maxlen = 64

    def text_segmentate(text, maxlen, seps='\n', strips=None):
        """将文本按照标点符号划分为若干个短句
        """
        text = text.strip().strip(strips)
        if seps and len(text) > maxlen:
            pieces = text.split(seps[0])
            text, texts = '', []
            for i, p in enumerate(pieces):
                if text and p and len(text) + len(p) > maxlen - 1:
                    texts.extend(
                        text_segmentate(text, maxlen, seps[1:], strips))
                    text = ''
                if i + 1 == len(pieces):
                    text = text + p
                else:
                    text = text + p + seps[0]
            if text:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
            return texts
        else:
            return [text]

    def split(text):
        """分割句子
        """
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        return text_segmentate(text, maxlen * 1.2, seps, strips)

    D = []
    labels = ['contradiction', 'entailment', 'neutral']
    with open(txt_path, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            if l['gold_label'] not in labels:
                continue
            text1 = split(l['sentence1'])[0]
            text2 = split(l['sentence2'])[0]
            label = labels.index(l['gold_label']) + 2
            D.append((text1, text2, label))
    return D


def readjson(path):
    f = open(path)
    data = json.load(f)
    return data


def BatchRename_filename(
        project_root_path="/content/Unet_multiloss/Unet/",
        dataset_relative_path="data_set",
        data_root_path="mito_testsample",
        data_sub_fol_list=["train_data", "train_label1", "train_label2"],
        fileprefix=["images", "mitos_3D", "mitos_3D"],
        start=-8):
    fullpaths = []
    for fol in data_sub_fol_list:
        fullpaths.append(
            os.path.join(project_root_path, dataset_relative_path,
                         data_root_path, fol))
    data_suffix = ".png"
    filename_list = files_with_extension(fullpaths[0], data_suffix)
    for fileIdx, filename in enumerate(filename_list):
        for folIdx, fullpath in enumerate(fullpaths):
            os.rename(
                os.path.join(fullpath, fileprefix[folIdx] + filename[start:]),
                os.path.join(
                    fullpath, data_sub_fol_list[folIdx] +
                    "{:04n}".format(fileIdx)) + filename[-4:])
    print("rename finished~~~~~~~~~")


def ZipPreprocess(zip_path, zip_name):
    d = datetime.now()
    zip_name = zip_name + str(d)
    save_file = os.path.join(zip_name + '.zip')
    if os.path.exists(save_file):
        os.remove()

    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    zipf = zipfile.ZipFile(zip_name + '.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(zip_path, zipf)
    zipf.close()


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file does not exist")


def delete_files_by_prefix(base_path, file_prefix):
    dir_list = glob.iglob(os.path.join(base_path, file_prefix + "*"))
    for path in dir_list:
        if os.path.exists(path):
            os.remove(path)


def delete_directory(folder_path):
    cleanfolder(folder_path)
    os.rmdir(folder_path)


def copy_file_recursive(source_path, target_path):

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if os.path.exists(source_path):
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, target_path)
                print(src_file)
    print('copy files finished!')


def SendRefinedPlot(opt_dict,
                    net_version,
                    acc_pic_save_path,
                    run_start_step,
                    predict_list,
                    start,
                    end,
                    email,
                    pwd,
                    log_path="",
                    suffix=""):
    plt.plot(predict_list[0])
    plt.plot(predict_list[1])
    opt_dict["loss_min"] = max(predict_list[0])
    name = 'Gold'
    plt.title(opt_dict["opt"][0]["name"] + "_" + name + "_" + str(start) +
              "_" + str(end))
    plt.ylabel('dice score')
    plt.xlabel('epoch')
    plt.legend(['soma', 'vessel'], loc='upper left')
    plt.savefig(
        path.join(
            acc_pic_save_path, opt_dict["opt"][0]["name"] + "_" + name + "_" +
            str(start) + "_" + str(end) + ".png"))
    plt.show()
    run_end_step = time()
    ImgPath = acc_pic_save_path
    extension = '.png'
    time_consume = run_end_step - run_start_step
    subject = net_version + suffix + "training from" + str(
        start) + " to " + str(end) + " finished!"
    SendMail(subject,
             ImgPath,
             extension,
             opt_dict,
             time_consume,
             email,
             pwd,
             files=files_with_extension(log_path, ".txt"))


def read_tsvfile(filename: str):
    """读取csv文件

    Args:
        filename (str): 文件路径

    Returns:
        rows (list): 每个元素对应csv的一行
        column_titles (list): 每个元素对应csv的列名称
    """
    # initializing the titles and rows list
    column_titles = []
    rows = []
    num_lines = len([1 for line in open(filename, "r")])
    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader((line.replace('\0', '')
                               for line in tqdm(csvfile,
                               desc=filename.split(
                                   os.sep)[-1]+" is filtering ",
                               total=num_lines)))
        # extracting field names through first row
        column_titles = next(csvreader)
        # extracting each data row one by one
        try:
            for row in tqdm(csvreader, desc=filename.split(os.sep)[-1]
                            + " is loading..", total=num_lines):
                rows.append(row)
        except Exception as e:
            traceback.print_exc()
        # get total number of rows
    #     print("Total no. of rows: %d" % (csvreader.line_num))

        # # printing the field names
    # print('Field names are:' + ', '.join(field for field in fields))

    # #  printing first 5 rows
    # print('\nFirst 5 rows are:\n')
    # for row in rows[:5]:
    #     # parsing each column of a row
    #     for col in row:
    #         print("%10s" % col, end=","),
    #     print('\n')
    return rows, column_titles


def max_length(file_path):
    num_length = [len(line) for line in open(file_path, "r", encoding="utf-8")]
    return max(num_length)


def save_table2csv(data, path, sortby=None, ascending=False):
    """save a table to csv file 

    Args:
        data (dict): key is column name and value is list of values belong to the column
        path (str): save path
        sortby ([type], optional): [description]. Defaults to None.
        ascending (bool, optional): [description]. Defaults to False.

    Example:
        nme = ["aparna", "pankaj", "sudhir", "Geeku"]
        deg = ["MBA", "BCA", "M.Tech", "MBA"]
        scr = [90, 40, 80, 98]
        dict = {'name': nme, 'degree': deg, 'score': scr}
        save_table2csv(dict, os.path.join(sys.path[0], "dict.csv"))
    """
    df = pd.DataFrame(data)
    if sortby is not None:
        df.sort_values([sortby], axis=0, ascending=ascending, inplace=True)
    # saving the dataframe
    df.to_csv(path)


if __name__ == '__main__':
    # ====================================================================================================================
    # 获取目录
    # sub_folders = get_all_sub_folders(
    #     path)
    # print(len(sub_folders))
    # json = readjsonline(
    #     path)
    # print(json)
    # =====================================================================================================================
    print(sort_dict_by_value({"a": 3, "b": 2, "c": 1}, increase=False))
    # ====================================================================================================================
    # 读csv文件
    # csv = read_tsvfile(
    #     path)
    # print(path)
    # ====================================================================================================================
    # 清空目录
    # cleanfolder(
    #     path)
    # print(
    #     file_size(
    #         path))
    # ====================================================================================================================
    # 获取指定文件夹同后缀名数据
    # files = files_with_extension(
    #     "/yuhang/draft/data/function_data_source/word_frequency/field/txtperline_2021-10-26/csv",
    #     'csv')
    # # print(files.index(path))
    # print(len(files))
    # print(files)
    # ====================================================================================================================
    # 递归复制文件
    # copy_file_recursive(
    #     "/yuhang/trained_model/bert-base-multilingual-cased-paswx-6lan/checkpoint-500",
    #     "/yuhang/trained_model/bert-base-multilingual-cased-paswx-6lan/multilingual/checkpoint-start"
    # )
    # ====================================================================================================================
    # 读文件
    # file_name = "F2"  # TP,T4,R7
    # txt = readtxt(
    #     "/yuhang/draft/data/function_data_source/knowledge_base/categories_detail/category_detail2/%s.txt" % file_name)
    # sum_count = 0
    # for ele in txt:
    #     sum_count += len(ele)
    # ngrams = read_pkl(
    #     "/yuhang/draft/Lab/new-word-discovery/vocab-raw/category_detail2_freq0/ngram/%s.pkl" % file_name)
    # ngrams_count = sum([ngrams[i]for i in ngrams])
    # print("总字数为%s字,ngram中所有key的频次累加和为%s" % (sum_count, ngrams_count))
