import math
import torch
from fastapi import APIRouter, UploadFile, File
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from src.exception.custom_exception import EmbeddingGenerationError, SimilarityMeasurementError
from src.common.base_response import BaseResponse
from src.common.base_response_status import BaseResponseStatus

router = APIRouter(prefix="/analysis", tags=["analysis"])

# 모델 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet50
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model = nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_embedding(file_obj):
    try:
        file_obj.seek(0)

        # 이미지를 RGB로 열기
        img = Image.open(file_obj).convert("RGB")

        # 전처리
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 모델에 넣어 특징 벡터 추출
        with torch.no_grad():
            embedding = model(input_tensor)

        # 결과 (batch, 2048, 1, 1) 형태 -> (2048,) 형태로 변환
        embedding = embedding.view(embedding.size(0), -1)
        return embedding.cpu().numpy()[0]
    except Exception:
        raise EmbeddingGenerationError()

def cosine_similarity(vec1, vec2):
    try:
        dot = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot / (norm_a * norm_b)
    except Exception:
        raise SimilarityMeasurementError()

@router.post("/compare_images")
async def compare_images(
        teacher_file: UploadFile = File(...),
        student_file: UploadFile = File(...)
):
    try:
        # 임베딩 추출
        teacher_emb = get_embedding(teacher_file.file)
        student_emb = get_embedding(student_file.file)

        # 코사인 유사도 계산
        similarity_score = cosine_similarity(teacher_emb, student_emb)

        accuracy = math.floor(float(similarity_score) * 100)

        return BaseResponse.base(BaseResponseStatus.SUCCESS, {"accuracy": accuracy})
    except SimilarityMeasurementError:
        return BaseResponse.base(BaseResponseStatus.SIMILARITY_MEASUREMENT_ERROR)
    except EmbeddingGenerationError:
        return BaseResponse.base(BaseResponseStatus.EMBEDDING_GENERATION_ERROR)
    except (Exception,):
        return BaseResponse.base(BaseResponseStatus.INTERNAL_SERVER_ERROR)
