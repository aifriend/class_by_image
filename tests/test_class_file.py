import os

import numpy as np
import pytest

from common.ClassFile import ClassFile


@pytest.fixture
def file_tree(tmp_path):
    (tmp_path / "sub").mkdir()
    files = {
        "a.jpg": b"123",
        "b.PDF": b"12345",
        "sub/c.txt": b"1",
        "sub/d.tar.gz": b"12",
        "empty.png": b"",
    }
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    return str(tmp_path)


def test_get_file_root_name_strips_all_extensions():
    assert ClassFile.get_file_root_name("archive.tar.gz") == "archive"
    assert ClassFile.get_file_root_name("noext") == "noext"


def test_file_base_name_uses_last_dot():
    assert ClassFile.file_base_name("a.b.txt") == "a.b"
    assert ClassFile.file_base_name("plain") == "plain"


def test_get_file_name():
    assert ClassFile.get_file_name(os.path.join("x", "y", "doc.pdf")) == "doc"


def test_list_files_and_has_file(file_tree):
    files = ClassFile.list_files(file_tree)
    assert len(files) == 5
    assert ClassFile.has_file(file_tree, "a.jpg")
    assert not ClassFile.has_file(file_tree, "zzz")


def test_list_files_ext_is_case_insensitive(file_tree):
    pdfs = ClassFile.list_files_ext(file_tree, "pdf")
    assert len(pdfs) == 1
    assert pdfs[0].endswith("b.PDF")


def test_filter_by_ext(file_tree):
    source = ClassFile.list_files(file_tree)
    assert len(ClassFile.filter_by_ext(source, ".txt")) == 1
    single = os.path.join(file_tree, "a.jpg")
    assert ClassFile.filter_by_ext(single, "jpg") == [single]


def test_filter_by_size_skips_empty(file_tree):
    source = ClassFile.list_files(file_tree)
    filtered = ClassFile.filter_by_size(source)
    assert all(os.path.getsize(f) > 0 for f in filtered)
    assert not any(f.endswith("empty.png") for f in filtered)


def test_filter_files_root():
    assert ClassFile.filter_files_root(["/x/a.jpg", "/y/b.png"]) == ["/x/a", "/y/b"]


def test_pickle_roundtrip(tmp_path):
    target = str(tmp_path / "data.pkl")
    ClassFile.list_to_file(["b", "a", "a"], target)
    assert ClassFile.file_to_list(target) == ["a", "b"]


def test_pickle_roundtrip_unsorted(tmp_path):
    target = str(tmp_path / "data.pkl")
    ClassFile.list_to_file_unsorted(["b", "a", "a"], target)
    assert ClassFile.file_to_list_unsorted(target) == ["b", "a", "a"]


def test_csv_to_numpy_image_returns_array(tmp_path):
    target = str(tmp_path / "img.csv")
    np.savetxt(target, np.arange(6).reshape(2, 3))
    result = ClassFile.csv_to_numpy_image(target)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)


def test_save_and_load_model(tmp_path):
    target = str(tmp_path / "model.pkl")
    ClassFile.save_model(target, {"weights": [1, 2, 3]})
    assert ClassFile.load_model(target) == {"weights": [1, 2, 3]}


def test_create_dir_idempotent(tmp_path):
    target = str(tmp_path / "new" / "nested")
    ClassFile.create_dir(target)
    ClassFile.create_dir(target)
    assert os.path.isdir(target)


def test_get_containing_dir_name():
    assert ClassFile.get_containing_dir_name("/a/b/c/file.jpg") == "c"
