from ..myLoader import MyLoader, MyLoaderCtx
import pytest
import os
import torch


@pytest.fixture
def file_path(request):
    # 获取当前测试文件所在目录
    current_dir = os.path.dirname(request.node.fspath)
    # 构建 'tests' 文件夹下某个文件的路径
    file_path = os.path.join(current_dir, "test_data.xlsx")
    return file_path


def test_loader(file_path):
    loader = MyLoader(MyLoaderCtx(file_path, 0.5, 2))
    loader.prepare_data()
    train_loader = loader.train_dataloader()
    for test in train_loader:
        assert len(test) == 2
    assert len(train_loader.dataset) == 4
