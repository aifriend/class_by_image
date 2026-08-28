from common.commonsLib import loggerElk, loggerFileAux


def test_logger_elk_defaults(monkeypatch):
    for var in ("ELK_ENABLED", "FILE_ENABLED", "LIBRARIES_LOG_LEVEL", "LOG_LEVEL", "LOG_FILE"):
        monkeypatch.delenv(var, raising=False)
    log = loggerElk("test-service")
    assert log.serviceName == "test-service"
    assert log.lib_lob_level == "ERROR"
    assert log.elkEnabled is False
    # log methods must not raise
    log.Information("info message")
    log.Debug("debug message")
    log.LogInput("input", {"k": 1})
    log.LogResult("result", [1, 2])
    log.Error("error message")


def test_logger_elk_file_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FILE_ENABLED", "True")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "app.log"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LIBRARIES_LOG_LEVEL", "DEBUG")
    log = loggerElk("test-service")
    log.Debug("visible debug")
    assert (tmp_path / "app.log").exists()


def test_logger_file_aux_respects_debug_mode():
    aux = loggerFileAux(debug_mode=True)
    aux.Log("INFO", "hello")
    assert aux.LOG_LIST == [{"level": "INFO", "message": "hello"}]

    aux_off = loggerFileAux(debug_mode=False)
    aux_off.Log("INFO", "hello")
    assert aux_off.LOG_LIST == []
