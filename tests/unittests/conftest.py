def pytest_configure(config):
    config.addinivalue_line("markers", "cache_dir(path): mark test to use specific cache directory")
