# Texture_project
본 프로젝트는 [threestudio](https://github.com/threestudio-project/threestudio) 기반으로 작성된 코드입니다.

## 1. 환경설정
```bash
conda create -n texture python=3.10 -y
conda activate texture
pip install -r requirements.txt
```

## 2. 다운로드
* [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter).
사전 학습된 `ip-adapter_sd15.bin` 모델을 사용합니다. `threestudio/models/guidance/adapter/models`에 배치해 주세요.
```bash
cd threestudio/models/guidance/adapter/models
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin
mkdir image_encoder
cd image_encoder
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin
```

### 3. 실행
```bash
python gradio.py
```
