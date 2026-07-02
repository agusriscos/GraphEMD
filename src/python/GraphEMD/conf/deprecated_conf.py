from CommonUtils.data import DictClass

class NoOverlapTransformConfig(DictClass):
    MIN_VAL: int = -500
    MAX_VAL: int = 500
    MIN_WINDOW_SIZE: int = 20
    MAX_WINDOW_SIZE: int = 50
    WINDOW_NUM: int = None
    WINDOW_MODE: int = 1