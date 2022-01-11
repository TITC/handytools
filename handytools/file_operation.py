from tqdm.std import tqdm
import os


class FileOperation(object):
    def __init__(self, *args):
        super(FileOperation, self).__init__(*args)

    def readlines(path, portion=1, end: int = None, show_bar=True, encoding="utf-8"):
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
                    [1 for line in open(path, "r", encoding=encoding)])
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


if __name__ == '__main__':
    # test readlines
    lines = FileOperation.readlines(
        path="/mnt/f/data/NLP/test_data/fiction/wjtx.txt")
    print(lines)
