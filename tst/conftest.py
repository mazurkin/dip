import os


# default: disable CUDA for all tests
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def pytest_collection_modifyitems(config, items):
    # Re-enable CUDA if any cuda-marked tests are collected
    if any('cuda' in item.keywords for item in items):
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
