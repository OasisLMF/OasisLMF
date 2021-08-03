from unittest import TestCase, main

from pandas import DataFrame

from oasislmf.pytools.getmodel.file_loader import FileLoader, read_csv


class TestFileLoader(TestCase):

    def setUp(self) -> None:
        self.test = FileLoader(file_path="one/two/three.csv", label="vulnerabilities")
        self.data = [{"one": 1, "two": 2}, {"one": 3, "two": 4}, {"one": 5, "two": 6}, {"one": 7, "two": 8}]

    def test___init__(self):
        self.assertEqual("one/two/three.csv", self.test.path)
        self.assertEqual("vulnerabilities", self.test.label)
        self.assertEqual("csv", self.test.extension)
        self.assertEqual(None, self.test._value)
        self.assertEqual(read_csv, self.test.get_read_function())

    def test_get_read_function(self):
        self.assertEqual(read_csv, self.test.get_read_function())

    def test_read(self):
        df = DataFrame(self.data)
        df.to_csv("./test.csv")
        test = FileLoader(file_path="./test.csv", label="vulnerabilities")
        self.assertEqual(DataFrame, type(test.value))
        self.assertEqual(self.data, list(test.value.T.to_dict().values()))

        test.clear_cache()
        self.assertEqual(None, test._value)

    def test_value_set(self):
        self.test.value = "something"
        self.assertEqual("something", self.test.value)


if __name__ == "__main__":
    main()
