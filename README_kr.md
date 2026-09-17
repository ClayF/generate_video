# Wan2.2 Generate Video API Client
[English README](README.md)

이 프로젝트는 RunPod의 generate_video 엔드포인트를 통해 **Wan2.2** 모델을 사용하여 이미지에서 비디오를 생성하는 Python 클라이언트를 제공합니다. 클라이언트는 base64 인코딩, LoRA 설정, 배치 처리 기능을 지원합니다.

[![Runpod](https://api.runpod.io/badge/wlsdml1114/generate_video)](https://console.runpod.io/hub/wlsdml1114/generate_video)

**Wan2.2**는 정적 이미지를 자연스러운 움직임과 사실적인 애니메이션을 가진 동적 비디오로 변환하는 고급 AI 모델입니다. ComfyUI 위에 구축되어 고품질 비디오 생성 기능을 제공합니다.

## 🎨 Engui Studio 통합

[![EnguiStudio](https://raw.githubusercontent.com/wlsdml1114/Engui_Studio/main/assets/banner.png)](https://github.com/wlsdml1114/Engui_Studio)

이 Wan2.2 클라이언트는 포괄적인 AI 모델 관리 플랫폼인 **Engui Studio**를 위해 주로 설계되었습니다. API를 통해 사용할 수 있지만, Engui Studio는 향상된 기능과 더 넓은 모델 지원을 제공합니다.

## ✨ 주요 기능

*   **Wan2.2 모델**: 고품질 비디오 생성을 위한 고급 Wan2.2 AI 모델로 구동됩니다.
*   **이미지-투-비디오 생성**: 정적 이미지를 자연스러운 움직임을 가진 동적 비디오로 변환합니다.
*   **Base64 인코딩 지원**: 이미지 인코딩/디코딩을 자동으로 처리합니다.
*   **LoRA 설정**: 향상된 비디오 생성을 위해 최대 4개의 LoRA 쌍을 지원합니다.
*   **배치 처리**: 단일 작업으로 여러 이미지를 처리할 수 있습니다.
*   **오류 처리**: 포괄적인 오류 처리 및 로깅 기능을 제공합니다.
*   **비동기 작업 관리**: 자동 작업 제출 및 상태 모니터링을 지원합니다.
*   **ComfyUI 통합**: 유연한 워크플로우 관리를 위해 ComfyUI 위에 구축되었습니다.

## 🚀 RunPod Serverless 템플릿

이 템플릿은 **Wan2.2**를 RunPod Serverless Worker로 실행하는 데 필요한 모든 구성 요소를 포함합니다.

*   **Dockerfile**: Wan2.2 모델 실행에 필요한 환경을 구성하고 모든 의존성을 설치합니다.
*   **handler.py**: RunPod Serverless용 요청을 처리하는 핸들러 함수를 구현합니다.
*   **entrypoint.sh**: Worker가 시작될 때 초기화 작업을 수행합니다.
*   **new_Wan22_api.json**: Wan2.2 이미지-투-비디오 생성을 위한 단일 워크플로우 파일로 최대 4개 LoRA 쌍까지 지원

## 📖 Python 클라이언트 사용법

### 기본 사용법

```python
from generate_video_client import GenerateVideoClient

# 클라이언트 초기화
client = GenerateVideoClient(
    runpod_endpoint_id="your-endpoint-id",
    runpod_api_key="your-runpod-api-key"
)

# 이미지에서 비디오 생성
result = client.create_video_from_image(
    image_path="./example_image.png",
    prompt="running man, grab the gun",
    negative_prompt="blurry, low quality, distorted",
    width=480,
    height=832,
    length=81,
    steps=10,
    seed=42,
    cfg=2.0
)

# 성공 시 결과 저장
if result.get('status') == 'COMPLETED':
    client.save_video_result(result, "./output_video.mp4")
else:
    print(f"오류: {result.get('error')}")
```

### LoRA 사용

```python
# LoRA 쌍 설정
lora_pairs = [
    {
        "high": "your_high_lora.safetensors",
        "low": "your_low_lora.safetensors",
        "high_weight": 1.0,
        "low_weight": 1.0
    }
]

# LoRA를 사용한 비디오 생성
result = client.create_video_from_image(
    image_path="./example_image.png",
    prompt="running man, grab the gun",
    negative_prompt="blurry, low quality, distorted",
    width=480,
    height=832,
    length=81,
    steps=10,
    seed=42,
    cfg=2.0,
    lora_pairs=lora_pairs
)
```

### 배치 처리

```python
# 여러 이미지 처리
batch_result = client.batch_process_images(
    image_folder_path="./input_images",
    output_folder_path="./output_videos",
    prompt="running man, grab the gun",
    negative_prompt="blurry, low quality, distorted",
    width=480,
    height=832,
    length=81,
    steps=10,
    seed=42,
    cfg=2.0
)

print(f"배치 처리 완료: {batch_result['successful']}/{batch_result['total_files']} 성공")
```

## 🔧 API 참조

### 입력

`input` 객체는 다음 필드를 포함해야 합니다. 이미지는 **경로, URL 또는 Base64** 중 하나의 방법으로 입력할 수 있습니다.

#### 이미지 입력 (하나만 사용)
| 매개변수 | 타입 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `image_path` | `string` | 아니오 | - | 입력 이미지의 로컬 경로 |
| `image_url` | `string` | 아니오 | - | 입력 이미지의 URL |
| `image_base64` | `string` | 아니오 | - | 입력 이미지의 Base64 인코딩된 문자열 |

#### LoRA 설정
| 매개변수 | 타입 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `lora_pairs` | `array` | 아니오 | `[]` | LoRA 쌍의 배열. 각 쌍은 `high`, `low`, `high_weight`, `low_weight`를 포함 |

**중요**: LoRA 모델을 사용하려면 RunPod 네트워크 볼륨의 `/loras/` 폴더에 LoRA 파일들을 업로드해야 합니다. `lora_pairs`의 LoRA 모델 이름은 `/loras/` 폴더의 파일명과 일치해야 합니다.

#### LoRA 쌍 구조
| 매개변수 | 타입 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `high` | `string` | 예 | - | High LoRA 모델 이름 |
| `low` | `string` | 예 | - | Low LoRA 모델 이름 |
| `high_weight` | `float` | 아니오 | `1.0` | High LoRA 가중치 |
| `low_weight` | `float` | 아니오 | `1.0` | Low LoRA 가중치 |

#### 비디오 생성 매개변수
| 매개변수 | 타입 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `prompt` | `string` | 예 | - | 생성할 비디오에 대한 설명 텍스트 |
| `negative_prompt` | `string` | 아니오 | - | 비디오에서 제외할 원하지 않는 요소에 대한 네거티브 프롬프트 |
| `seed` | `integer` | 아니오 | `42` | 비디오 생성을 위한 랜덤 시드 |
| `cfg` | `float` | 아니오 | `2.0` | 생성을 위한 CFG 스케일 |
| `width` | `integer` | 아니오 | `480` | 출력 비디오의 픽셀 단위 너비 |
| `height` | `integer` | 아니오 | `832` | 출력 비디오의 픽셀 단위 높이 |
| `length` | `integer` | 아니오 | `81` | 생성할 비디오의 길이 |
| `steps` | `integer` | 아니오 | `10` | 디노이징 스텝 수 |
| `context_overlap` | `integer` | 아니오 | `48` | 컨텍스트 오버랩 값 |

#### 프롬프트 확장 (선택)
| 매개변수 | 타입 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `prompt_expansion` | `boolean` 또는 `object` | 아니오 | `false` | `true`이면 워커 기본값으로 프롬프트 확장을 켭니다. 객체로 세부 설정을 지정할 수 있습니다 (아래 필드). `expand_prompt: true`도 같은 의미입니다. |
| `prompt_expansion.language` | `string` | 아니오 | `zh` | 확장된 프롬프트의 언어: `zh`, `en`, `auto`. Wan2.2는 중국어 프롬프트를 훨씬 잘 따르므로 `zh`가 기본값입니다. |
| `prompt_expansion.model` | `string` | 아니오 | 첫 번째 로컬 Qwen GGUF | 로컬 모델은 파일명, `LLM/<파일>.gguf` 또는 `/runpod-volume/LLM/<파일>.gguf`, DashScope 클라우드 모델은 `qwen3.7-plus` 같은 이름 (엔드포인트에 `DASHSCOPE_API_KEY` 필요). |
| `prompt_expansion.device` | `string` | 아니오 | `GPU` | 로컬 GGUF 추론 장치: `GPU` 또는 `CPU`. |
| `prompt_expansion.mmproj` | `string` | 아니오 | `(Auto-detect)` | 로컬 모델의 mmproj 선택. |
| `prompt_expansion.max_retries` | `integer` | 아니오 | `3` | 클라우드 모델 재시도 횟수 (1-10). |
| `prompt_expansion.save_tokens` | `boolean` | 아니오 | `true` | 클라우드 모델로 보내기 전에 이미지를 축소합니다. |
| `expand_only` | `boolean` | 아니오 | `false` | 프롬프트 확장만 실행하고 텍스트를 반환합니다 (비디오 생성 없음). 렌더링 전에 프롬프트를 미리 보고 수정할 때 유용합니다. |

자세한 내용은 [프롬프트 확장](#️-프롬프트-확장) 섹션을 참고하세요.

**요청 예시:**

#### 1. 기본 생성 (LoRA 없음)
```json
{
  "input": {
    "prompt": "running man, grab the gun",
    "negative_prompt": "blurry, low quality, distorted",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "seed": 42,
    "cfg": 2.0,
    "width": 480,
    "height": 832,
    "length": 81,
    "steps": 10
  }
}
```

#### 2. LoRA 쌍 사용
```json
{
  "input": {
    "prompt": "running man, grab the gun",
    "negative_prompt": "blurry, low quality, distorted",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "seed": 42,
    "cfg": 2.0,
    "width": 480,
    "height": 832,
    "lora_pairs": [
      {
        "high": "your_high_lora.safetensors",
        "low": "your_low_lora.safetensors",
        "high_weight": 1.0,
        "low_weight": 1.0
      }
    ]
  }
}
```

#### 3. 여러 LoRA 쌍 (최대 4개)
```json
{
  "input": {
    "prompt": "running man, grab the gun",
    "negative_prompt": "blurry, low quality, distorted",
    "image_path": "/my_volume/image.jpg",
    "seed": 42,
    "cfg": 2.0,
    "width": 480,
    "height": 832,
    "lora_pairs": [
      {
        "high": "lora1_high.safetensors",
        "low": "lora1_low.safetensors",
        "high_weight": 1.0,
        "low_weight": 1.0
      },
      {
        "high": "lora2_high.safetensors",
        "low": "lora2_low.safetensors",
        "high_weight": 1.0,
        "low_weight": 1.0
      }
    ]
  }
}
```

#### 4. URL 이미지 입력
```json
{
  "input": {
    "prompt": "running man, grab the gun",
    "negative_prompt": "blurry, low quality, distorted",
    "image_url": "https://example.com/image.jpg",
    "seed": 42,
    "cfg": 2.0,
    "width": 480,
    "height": 832,
    "context_overlap": 48
  }
}
```

#### 5. 프롬프트 확장 후 생성
```json
{
  "input": {
    "prompt": "the camera slowly pushes in as she looks up",
    "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "seed": 42,
    "cfg": 2.0,
    "width": 480,
    "height": 832,
    "prompt_expansion": { "language": "zh", "device": "GPU" }
  }
}
```

#### 6. 프롬프트 확장만 (미리보기, 비디오 없음)
```json
{
  "input": {
    "prompt": "the camera slowly pushes in as she looks up",
    "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "width": 480,
    "height": 832,
    "expand_only": true,
    "prompt_expansion": { "language": "en" }
  }
}
```

### 출력

#### 성공

작업이 성공하면 생성된 비디오가 Base64로 인코딩된 JSON 객체를 반환합니다.

| 매개변수 | 타입 | 설명 |
| --- | --- | --- |
| `video` | `string` | Base64로 인코딩된 비디오 파일 데이터입니다. |
| `expanded_prompt` | `string` | 프롬프트 확장을 켰을 때만: 실제로 Wan2.2에 전달된 확장 프롬프트. |
| `prompt` | `string` | 프롬프트 확장을 켰을 때만: 요청에 보낸 원본 프롬프트. |

`expand_only: true`인 경우 출력에는 `expanded_prompt`와 `prompt`만 포함됩니다.

**성공 응답 예시:**

```json
{
  "video": "data:video/mp4;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
}
```

#### 오류

작업이 실패하면 오류 메시지를 포함한 JSON 객체를 반환합니다.

| 매개변수 | 타입 | 설명 |
| --- | --- | --- |
| `error` | `string` | 발생한 오류에 대한 설명입니다. |

**오류 응답 예시:**

```json
{
  "error": "Video not found."
}
```

## 🛠️ 직접 API 사용법

1.  이 저장소를 기반으로 RunPod에서 Serverless Endpoint를 생성합니다.
2.  빌드가 완료되고 엔드포인트가 활성화되면 위의 API 참조에 따라 HTTP POST 요청을 통해 작업을 제출합니다.

### 📁 네트워크 볼륨 사용

Base64로 인코딩된 파일을 직접 전송하는 대신 RunPod의 Network Volumes를 사용하여 대용량 파일을 처리할 수 있습니다. 이는 특히 대용량 이미지 파일과 LoRA 모델을 다룰 때 유용합니다.

1.  **네트워크 볼륨 생성 및 연결**: RunPod 대시보드에서 Network Volume(예: S3 기반 볼륨)을 생성하고 Serverless Endpoint 설정에 연결합니다.
2.  **파일 업로드**: 사용하려는 이미지 파일과 LoRA 모델을 생성된 Network Volume에 업로드합니다.
3.  **파일 구성**: 
    - 입력 이미지는 Network Volume 내 어디든 배치할 수 있습니다
    - LoRA 모델 파일은 Network Volume 내의 `/loras/` 폴더에 배치해야 합니다
4.  **경로 지정**: API 요청 시 Network Volume 내의 파일 경로를 지정합니다:
    - `image_path`의 경우: 이미지 파일의 전체 경로 사용 (예: `"/my_volume/images/portrait.jpg"`)
    - LoRA 모델의 경우: 파일명만 사용 (예: `"my_lora_model.safetensors"`) - 시스템이 자동으로 `/loras/` 폴더에서 찾습니다

## 🔧 클라이언트 메서드

### GenerateVideoClient 클래스

#### `__init__(runpod_endpoint_id, runpod_api_key)`
RunPod 엔드포인트 ID와 API 키로 클라이언트를 초기화합니다.

#### `create_video_from_image(image_path, prompt, width, height, length, steps, seed, cfg, context_overlap, lora_pairs, negative_prompt)`
단일 이미지에서 비디오를 생성합니다.

**매개변수:**
- `image_path` (str): 입력 이미지 경로
- `prompt` (str): 비디오 생성을 위한 텍스트 프롬프트
- `negative_prompt` (str): 원하지 않는 요소를 제외하기 위한 네거티브 프롬프트 (기본값: None)
- `width` (int): 출력 비디오 너비 (기본값: 480)
- `height` (int): 출력 비디오 높이 (기본값: 832)
- `length` (int): 프레임 수 (기본값: 81)
- `steps` (int): 디노이징 스텝 수 (기본값: 10)
- `seed` (int): 랜덤 시드 (기본값: 42)
- `cfg` (float): CFG 스케일 (기본값: 2.0)
- `context_overlap` (int): 컨텍스트 오버랩 (기본값: 48)
- `lora_pairs` (list): LoRA 설정 쌍 (기본값: None)

#### `batch_process_images(image_folder_path, output_folder_path, valid_extensions, ...)`
폴더 내 여러 이미지를 처리합니다.

**매개변수:**
- `image_folder_path` (str): 이미지가 포함된 폴더 경로
- `output_folder_path` (str): 출력 비디오를 저장할 경로
- `valid_extensions` (tuple): 유효한 이미지 확장자 (기본값: ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
- 기타 매개변수는 `create_video_from_image`와 동일

#### `save_video_result(result, output_path)`
비디오 결과를 파일로 저장합니다.

**매개변수:**
- `result` (dict): 작업 결과 딕셔너리
- `output_path` (str): 비디오 파일을 저장할 경로

## ✍️ 프롬프트 확장

Wan2.2는 짧은 단어 몇 개보다 길고 구체적인 프롬프트, 특히 중국어 프롬프트에 훨씬 잘 반응합니다. 프롬프트 확장을 켜면 [ComfyUI-MultiModal-Prompt-Nodes](https://github.com/kantan-kanto/ComfyUI-MultiModal-Prompt-Nodes)의 **Wan Video Prompt Generator** 노드가 워커에서 시작 프레임(핸들러가 리사이즈/크롭한 뒤의 이미지, 즉 비디오 모델이 실제로 보는 이미지)과 사용자의 모션 프롬프트를 보고 완전한 image-to-video 프롬프트를 작성합니다. 확장된 프롬프트는 텍스트 인코더로 바로 전달되며 `expanded_prompt`로도 반환됩니다.

### 연결 구조

```
244 LoadImage ─► 171 Resize ─► 900 WanVideoPromptGenerator ─► 901 GVPromptOutput ─► 135 WanVideoTextEncode ─► 샘플러
                                     ▲ prompt (사용자 텍스트)              │
                                                                          └─► 작업 히스토리에 기록 → "expanded_prompt"
```

*   `new_Wan22_expand_api.json`, `new_Wan22_flf2v_expand_api.json`은 기본 워크플로우에 **900**, **901** 노드를 추가한 것이며, **135** 노드의 `positive_prompt`는 문자열 대신 링크가 됩니다. 기본 워크플로우를 수정한 뒤에는 `python tools/build_expand_workflows.py`로 다시 생성하세요 (`--check`로 검증).
*   `expand_only: true`이면 244/171/235/236/900/901 노드만 실행하므로 디퓨전 모델은 전혀 로드되지 않습니다. 워밍업된 워커에서는 GPU 기준 수 초면 끝납니다.
*   기본 모델은 [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)의 **Qwen3-VL-8B-Instruct (Q8_0)**와 mmproj(합쳐서 약 9.4 GB)입니다. RunPod 빌드 시간 제한 때문에 이미지에 포함하지 않고, `entrypoint.sh`가 첫 시작 때 `/runpod-volume/LLM/`(네트워크 볼륨 권장, 없으면 컨테이너 디스크)에 내려받습니다. 직접 빌드한다면 `--build-arg BAKE_PROMPT_LLM=1`로 이미지에 포함할 수 있습니다. Qwen3-VL 비전 지원을 위해 `llama-cpp-python`의 CUDA 빌드([JamePeng 포크](https://github.com/JamePeng/llama-cpp-python))를 사용합니다. 노드는 답변 직후 LLM을 언로드하므로 샘플링 중 Wan2.2와 VRAM을 두고 경쟁하지 않습니다.

### 엔드포인트 설정

| 환경 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `PROMPT_EXPANSION_LANGUAGE` | `zh` | 요청에 언어가 없을 때의 기본 출력 언어 |
| `PROMPT_EXPANSION_DEVICE` | `GPU` | 로컬 GGUF 추론 기본 장치 |
| `PROMPT_EXPANSION_MODEL` | `models/LLM`, 그다음 `/runpod-volume/LLM`의 첫 번째 Qwen GGUF | 기본 모델 (`<파일>.gguf`, `LLM/<파일>.gguf`, `/runpod-volume/LLM/<파일>.gguf` 또는 DashScope 모델 이름) |
| `PROMPT_LLM_URL` / `PROMPT_MMPROJ_URL` | Qwen3-VL-8B-Instruct Q8_0 | 모델이 없을 때 `entrypoint.sh`가 내려받는 파일 |
| `PROMPT_LLM_AUTO_DOWNLOAD` | `1` | `0`이면 시작 시 다운로드 생략 |
| `DASHSCOPE_API_KEY` | – | 시작 시 노드의 `api_key.txt`에 기록되어 클라우드 Qwen 모델을 `prompt_expansion.model`로 사용할 수 있게 합니다 |

엔드포인트 환경 변수 `PROMPT_LLM_URL` / `PROMPT_MMPROJ_URL`로 모델을 바꿀 수 있습니다 (모델과 mmproj는 같은 폴더에, 파일명에는 mmproj 자동 감지를 위해 `Qwen3VL` 같은 패밀리 접두사 유지). 빌드 인자 `BAKE_PROMPT_LLM=1`은 빌드 시 모델을 이미지에 포함합니다. `LLAMA_CPP_PYTHON_RELEASE` / `LLAMA_CPP_PYTHON_WHEEL_VERSION`은 프리빌드 CUDA 휠을 고정합니다. 네트워크 볼륨의 `/LLM/`에 넣은 GGUF도 노드에서 보이며 `prompt_expansion.model: "/runpod-volume/LLM/<파일>.gguf"`로 선택합니다.

### 참고

*   프롬프트 확장은 명시적으로 요청할 때만 켜지므로 기존 클라이언트는 보낸 프롬프트를 그대로 사용합니다.
*   ComfyUI는 노드 결과를 입력 기준으로 캐시합니다. base64/URL 이미지는 작업마다 새 경로를 받아 캐시되지 않지만, 네트워크 볼륨의 동일한 `image_path`와 동일한 프롬프트는 워밍업된 워커에서 이전 확장 결과를 재사용합니다.
*   LLM 노드 오류(mmproj 누락, 노드 목록에 없는 모델 이름, 클라우드 키 누락 등)는 일반적인 "video not found" 대신 작업의 `error` 필드로 반환됩니다.

## 🔧 Wan2.2 워크플로우 구성

이 템플릿은 **Wan2.2**를 위한 네 개의 ComfyUI API 형식 워크플로우를 포함하며, 핸들러가 요청마다 하나를 선택합니다:

| 파일 | 프레임 | 프롬프트 확장 |
| --- | --- | --- |
| `new_Wan22_api.json` | 시작 프레임 | – |
| `new_Wan22_flf2v_api.json` | 시작 + 끝 프레임 | – |
| `new_Wan22_expand_api.json` | 시작 프레임 | ✅ |
| `new_Wan22_flf2v_expand_api.json` | 시작 + 끝 프레임 | ✅ |

워크플로우는 ComfyUI를 기반으로 하며 Wan2.2 처리를 위한 모든 필요한 노드를 포함합니다:
- 프롬프트를 위한 T5 텍스트 인코딩 (선택적으로 프롬프트 확장 노드가 공급)
- VAE 로딩 및 처리
- 비디오 생성을 위한 WanVideo ImageToVideo Encode 노드
- LoRA 로딩 및 적용 노드 (WanVideoLoraSelectMulti)
- 이미지 리사이즈 및 CLIP 비전 처리 노드

### 테스트

`python -m unittest discover -s tests -v`는 ComfyUI나 RunPod 없이 요청 파싱, 그래프 연결, 출력 처리를 검증합니다. `MULTIMODAL_NODE_SRC=/path/to/ComfyUI-MultiModal-Prompt-Nodes`를 지정하면 900번 노드의 입력을 실제 노드 정의와도 대조합니다.

## 🙏 Wan2.2 소개

**Wan2.2**는 자연스러운 움직임과 사실적인 애니메이션을 가진 고품질 비디오를 생성하는 최첨단 AI 모델입니다. 이 프로젝트는 Wan2.2 모델의 쉬운 배포와 사용을 위한 Python 클라이언트와 RunPod 서버리스 템플릿을 제공합니다.

### Wan2.2의 주요 특징:
- **고품질 출력**: 우수한 시각적 품질과 부드러운 움직임을 가진 비디오 생성
- **자연스러운 애니메이션**: 정적 이미지에서 사실적이고 자연스러운 움직임 생성
- **LoRA 지원**: 세밀한 조정된 비디오 생성을 위한 LoRA (Low-Rank Adaptation) 지원
- **ComfyUI 통합**: 유연한 워크플로우 관리를 위한 ComfyUI 기반 구축
- **사용자 정의 가능한 매개변수**: 비디오 생성 매개변수의 완전한 제어

## 🙏 원본 프로젝트

이 프로젝트는 다음 원본 저장소를 기반으로 합니다. 모델과 핵심 로직에 대한 모든 권리는 원본 작성자에게 있습니다.

*   **Wan2.2:** [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
*   **ComfyUI:** [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
*   **ComfyUI-WanVideoWrapper** [https://github.com/kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

## 📄 라이선스

원본 Wan2.2 프로젝트는 해당 라이선스를 따릅니다. 이 템플릿도 해당 라이선스를 준수합니다.
