from tqdm.std import tqdm
import os
import pickle
import traceback
import shutil
import glob
import jsonlines
import json


class File(object):
    def __init__(self, *args):
        super(File, self).__init__(*args)

    def readlines(self, path, portion=1, end: int = None, show_bar=True, encoding="utf-8"):
        """read file and return a list of lines
        Args:
            path (string): file path
            portion (int, optional): specify the ratio of lines to be read. Defaults to 1.
            end (int, optional): specify which line to stop read. Defaults to None.
            show_bar (bool, optional): display the read progress or not. Defaults to True.
            encoding (str,optional): encoding format.Defaults to "utf-8".
        Returns:
            list: each elements is a row
        Examples:
            >>> lines = File.readlines("test.txt")
            >>> print(lines)
            ['line1', 'line2', 'line3']
        """
        data_file = []
        with open(path, "r+", encoding=encoding) as f:
            if end is None:
                num_lines = len(
                    [1 for _ in open(path, "r", encoding=encoding)])
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

    def glob_read(self, path, read_fun, stop_i=None,  show_bar=False):
        """read all files in path by glob

        Args:
            path (str): absolute path
            read_fun ([type]): different read function based on file type
            stop_i (int, optional): stop read at file i. Defaults to None.
            show_bar (bool, optional): display read process or not . Defaults to False.

        Returns:
            text_list (list): a list of lines in all files
            pathlist (list): a list of absolute file path

        Examples:
            >>> text_list, pathlist = glob_read("/home/directory/test", read_fun=read_txt)
            >>> print(text_list)
            ['line1', 'line2', 'line3']
        """
        text_list = []
        pathlist = glob.glob(path)
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

    def rjsonl(self, path, show_bar=False):
        """read jsonlines file

        Args:
            path (str): file path
            show_bar (bool, optional): display read process or not . Defaults to False.

        Returns:
            each time yield a json object

        Examples:
            >>> for item in File.rjsonl("test.jsonl"):
            >>>     print(item) 
            {'a': 1}
            {'b': 2}
            {'c': 3}
        """
        with open(path, "r+", encoding="utf8") as f:
            lines = [1 for _ in open(path, "r", encoding="utf-8")]
            if show_bar:
                total_nums = len(lines)
                w = tqdm(lines, total=total_nums, desc=u'已加载0个text')
            else:
                w = lines
            reader = jsonlines.Reader(f)
            for i, _ in enumerate(w):
                try:
                    item = reader.read()
                    yield item
                except Exception:
                    traceback.print_exc()
                if show_bar:
                    w.set_description(u'已加载%s个text' % str(i+1))
        f.close()

    def mkdirp(self, dir_path):
        """make directory if dir_path not exist

        Args:
            dir_path (string): directory absolute path

        Returns:
            state: if successfully created directory return True,
                    nontheless return False
        Examples:
            >>> File.mkdirp("/home/directory/test")
            True
        """
        state = True
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except Exception:
            state = False
        return state

    def save2pkl(self, obj: dict, path: str):
        """save object to pickle file

        Args:
            obj (dict): object to be saved
            path (str): file path
        Returns:
            state: if successfully created directory return True,
                    nontheless return False
        Examples:
            >>> File.save2pkl({"a": 1}, "test.pkl")
            True
        """
        try:
            self.mkdirp(os.path.dirname(path))
            f = open(path, "wb")
            pickle.dump(obj, f)
            f.close()
            return True
        except Exception:
            traceback.print_exc()
            return False

    def files_with_extension(self, path: str, extension: str):
        """get files by filename extension

        Args:
            path ([type]): [description]
            extension ([type]): [description]

        Returns:
            file_list: file name list
        """
        file_list = [
            file for file in os.listdir(path)
            if file.lower().endswith(extension)
        ]
        return file_list

    def parent_dir(self, path: str, layers: int = 1):
        """get parent directory

        Args:
            path (str): file path
            layers (int, optional): layers of parent directory. Defaults to 1.

        Returns:
            path: parent directory
        """
        dirname = os.path.dirname
        for _ in range(layers):
            path = dirname(path)
        return path

    def newpth(self, path, subfolder="", filename: str = ""):
        """get new path

        Args:
            path (str): absolute path
            subfolder (str, optional): sub-folder inside this path . Defaults to "".
            filename (str, optional): filename inside sub-folder. Defaults to "".

        Returns:
            str: absolute path /path/subfolder/filename

        Examples:
            >>> newpth("/home/directory/test", "subfolder", "filename")
            /home/directory/test/subfolder/filename
        """
        return os.path.join(self.parent_dir(path, 1), subfolder, filename)

    def get_all_sub_folders(self, path):
        """get all sub folders by path

        Args:
            path (str): target path

        Returns:
            list: sub folder list

        Examples:
            >>> get_all_sub_folders("/home/directory/test")
            ['test1', 'test2']
        """
        sub_directory = []
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isdir(file_path):
                sub_directory.append(filename)
        return sub_directory

    def generate_cleanfolder(self, path: str):
        """clean/generate a directory by path recursively

        Args:
            path (str): target path

        Returns:
            state: if successfully created directory return True,

        Examples:
            >>> generate_cleanfolder("/home/directory/test")
            True
        """
        folder_paths = []
        if isinstance(path, list):
            folder_paths.extend(path)
        else:
            folder_paths.append(path)
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
                    return False
        return True

    def generate_emptyfolder_bylist(self, root_path, folders_list):
        """generate empty folder by list

        Args:
            root_path (str): absolute path
            folders_list (list): folder list

        Examples:
            >>> generate_emptyfolder_bylist("/home/directory/test", ["test1", "test2"])
        """
        floder_list = []
        for fol in folders_list:
            floder_list.append(os.path.join(root_path, fol))
        self.generate_cleanfolder(floder_list)

    def file_size(self, path):
        """give file absolute path, return file size

        Args:
            path (str): absolute path

        Returns:
            int: file size in bytes
            int: file size in kb
            int: file size in mb

        Examples:
            >>> file_size("/home/directory/test.txt")
            (9, 0.009, 0.0009)
        """
        size = os.path.getsize(path)
        return size, round(size / (1024**2),
                           2), round(size / (1024**2),
                                     2), round(size / (1024**3), 2)

    def savelist(self, obj, path):
        """save list to file

        Args:
            obj (list): a list to be saved
            path (str): file path

        Returns:
            state: if successfully created directory return True

        Examples:
            >>> savelist([1,2,3], "test.txt")
            True
        """
        try:
            self.mkdirp(self.parent_dir(path, 1))
            with open(path, 'w') as f:
                for item in tqdm(obj, desc=path.split(os.sep)[-1] + " is saving..."):
                    f.write("%s\n" % item)
            return True
        except Exception as ex:
            traceback.print_exc()
            return False

    def append_list2file(self, path, obj, show_bar=False):
        """append list to file

        Args:
            path (str): absolute path
            obj (list): a list to be saved
            show_bar (bool, optional): [description]. Defaults to False.

        Returns:
            state: if successfully created directory return True

        Examples:
            >>> append_list2file("test.txt", [1,2,3])
            True
            >>> append_list2file("test.txt", [4,5,6], True)
            True
        """
        self.mkdirp(os.path.dirname(path))
        try:
            with open(path, 'a+') as f:
                if show_bar:
                    for item in tqdm(obj, desc=path.split(os.sep)[-1] + " is saving..."):
                        f.write("%s\n" % item)
                else:
                    for item in obj:
                        f.write("%s\n" % item)
            return True
        except Exception as ex:
            traceback.print_exc()
            return False

    def readjson(path, encoding='utf-8'):
        """read json file

        Args:
            path (str): absolute path
            encoding (str, optional): [description]. Defaults to 'utf-8'.

        Returns:
            dict: json data

        Examples:
            >>> readjson("test.json")
            {'a': 1, 'b': 2}
        """
        f = open(path)
        data = json.load(f, encoding=encoding)
        return data


class Dict(object):
    def __init__(self, *args):
        super(Dict, self).__init__(*args)

    def union_dict(self, dict1, dict2):
        """combine two dicts

        Args:
            dict1 (dict): only allow dict which value is int
            dict2 (dict): only allow dict which value is int

        Returns:
            dict2: combined dict
        Examples:
            >>> d = Dict()
            >>> d.union_dict({"a": 1}, {"b": 2})
            {'a': 1, 'b': 2}
        """
        for key in dict1.keys():
            if dict2.get(key) != None:
                dict2[key] += dict1[key]
            else:
                dict2[key] = dict1[key]
        return dict2

    def split_dict(self, dictionary: dict, split_nums: int):
        """split dict into several parts

        Args:
            dictionary (dict): a dict to be split
            split_nums (int): split nums

        Returns:
            list: each element is a dict

        Examples:
            >>> d = Dict()
            >>> d.split_dict({"a": 1, "b": 2, "c": 3}, 2)
            [{'a': 1, 'b': 2}, {'c': 3}]
        """
        dict_lengths = len(dictionary)
        batch_size = dict_lengths // split_nums
        batch_dict = []
        for n in range(split_nums + 1):
            cur_idx = batch_size * n
            end_idx = batch_size * (n + 1)
            cur_batch = dict(list(dictionary.items())[cur_idx: end_idx])
            batch_dict.append(cur_batch)
        return batch_dict

    def readpkl(self, dict_path):
        """read pickle file

        Args:
            dict_path (str): pickle file path

        Returns:
            dict_object (dict): pickle file object

        Examples:
            >>> d = Dict()
            >>> d.readpkl("test.pkl")
            {'a': 1, 'b': 2}
        """
        with open(dict_path, "rb") as f:
            dict_object = pickle.load(f)
        return dict_object

    def viw_pkl(self, path, start=0, end=10):
        """view dict in pickle file from start to end

        Args:
            path (str): absolute path
            start (int, optional): start index of dict. Defaults to 0.
            end (int, optional): end index of dict. Defaults to 10.

        Returns:
            result (dict): a small dict

        Examples:
            >>> d = Dict()
            >>> d.viw_pkl("/home/directory/test.pkl", 0, 10)
            {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10}
        """
        n_pkl = []
        with open(path, "rb") as f:
            dict_object = pickle.load(f)
            result = dict(list(dict_object.items())[start: end])
        return result

    def get_keys(slef, val, obj: dict):
        """get keys by value in dict

        Args:
            val ([type]): value
            obj (dict): dict

        Returns:
            list: keys\

        Examples:
            >>> d = Dict()
            >>> d.get_keys(1, {"a": 1, "b": 2})
            ['a']
        """
        return [k for k, v in obj.items() if v == val]

    def sort_dict_by_value(self, d, increase=True):
        """sort dict by value

        Args:
            d (dict): dict to be sorted
            increase (bool, optional): increase sort or decrease sort. Defaults to True.

        Returns:
            [type]: [description]

        Examples:
            >>> d = Dict()
            >>> d.sort_dict_by_value({"a": 1, "b": 2, "c": 3}, increase=False)
            [{'c': 3}, {'b': 2}, {'a': 1}]
        """
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=not increase))


class String(object):
    def __init__(self, *args):
        super(String, self).__init__(*args)


if __name__ == '__main__':
    # test readlines
    file = File()
    data = file.rjsonl(
        path="/mnt/f/git_repository/handytools/handytools/test.jsonl", show_bar=True)
    for datum in data:
        pass
