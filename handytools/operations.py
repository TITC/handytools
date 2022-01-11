from tqdm.std import tqdm
import os
import pickle
import traceback


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


class String(object):
    def __init__(self, *args):
        super(String, self).__init__(*args)


if __name__ == '__main__':
    # test readlines
    file = File()
    lines = file.readlines(
        path="/mnt/f/data/NLP/test_data/fiction/wjtx.txt")
    print(lines)
