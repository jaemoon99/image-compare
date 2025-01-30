from pydantic import BaseModel
from typing import Generic, TypeVar, Optional

from src.common.base_response_status import BaseResponseStatus

T = TypeVar("T")

class BaseResponse(BaseModel, Generic[T]):
    is_success: bool
    message: str
    code: int
    data: Optional[T]

    @classmethod
    def base(cls, status: BaseResponseStatus, result: T = None):
        return cls(
            is_success=status.is_success,
            message=status.message,
            code=status.code,
            data=result if result is not None else []
        )
