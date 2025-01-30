from enum import Enum

class BaseResponseStatus(Enum):
    SUCCESS = (True, 200, "요청에 성공하였습니다.")
    INTERNAL_SERVER_ERROR = (False, 500, "서버에 오류가 발생했습니다.")
    SIMILARITY_MEASUREMENT_ERROR = (False, 6002, "벡터 계산 중 오류가 발생했습니다.")
    EMBEDDING_GENERATION_ERROR = (False, 6003, "임베딩 생성 중 오류가 발생했습니다.")

    def __init__(self, is_success: bool, code: int, message: str):
        self.is_success = is_success
        self.code = code
        self.message = message
