from tqdm.std import tqdm
import os
import pickle
import traceback
import shutil


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

    def mkdirp(self, dir_path):
        """make directory if dir_path not exist

        Args:
            dir_path (string): directory absolute path

        Returns:
            status: if successfully created directory return True,
                    nontheless return False
        Examples:
            >>> File.mkdirp("/home/directory/test")
            True
        """
        status = True
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except Exception:
            status = False
        return status

    def save2pkl(self, obj: dict, path: str):
        """save object to pickle file

        Args:
            obj (dict): object to be saved
            path (str): file path
        Returns:
            status: if successfully created directory return True,
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

    def parent_dir(path: str, layers: int = 1):
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
            status: if successfully created directory return True,

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
    lines = file.readlines(
        path="/mnt/f/data/NLP/test_data/fiction/wjtx.txt")
    print(lines)
