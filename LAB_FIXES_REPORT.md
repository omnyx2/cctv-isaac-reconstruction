# lab-fixes 보고서: 강의 실습용 수정과 master 병합 검증

2026-10-11 · DGX Spark (GB10, aarch64) · Isaac Sim 5.1

어린이 실습 강의(CCTV 영상 → YOLO → Isaac Sim 3D 재현 + VLM)를 Linux DGX Spark에서 돌리며 필요했던 수정 4가지와,
그 위에 master `f1d0ebc`(scene-logic 카메라 보정, 차선 페인트)를 병합하고 다시 검증한 결과입니다.

## 1. 수정 내용 (6개 파일, +34 / −4)

| 수정 | 파일 | 이유 |
| --- | --- | --- |
| Linux에서 Isaac Sim 자체 Python 인식 | `session.py` (`running_in_own_python`), `setup_env.py` | Linux에서는 익스텐션이 Isaac Sim Python 안에서 도는지 판별이 안 돼 시작 단계가 실패 |
| YOLO 프로세스에 익스텐션 패키지 경로 전달 | `perception/perception_worker.py` | 별도 프로세스에서 `.pydeps` 패키지(cv2, torch 등)를 못 찾음 |
| VLM 모델 설정 `vlm_model` | `extension.py`, `config/extension.toml` | 실행 인자 `--/exts/cctv.dual_tracking/vlm_model=...`로 모델 고정 |
| Ollama 문맥 길이 `num_ctx: 8192` | `perception/vlm/client.py` | 지정하지 않으면 qwen3-vl이 262k 문맥을 잡아 메모리 45GB 사용 (8k면 19GB) |

## 2. master 병합

- 커밋: `002b035` (위 수정) + `65ef6ea` (origin/master `f1d0ebc` 병합).
- 충돌은 `session.py`의 `runnable_configs` 하나. master도 "영상이 있는 시나리오만 표시"로 같은 수정을 했으므로 master 쪽을 채택.
- 나머지 수정은 변경 없이 병합됨. 강의 스크립트가 쓰는 익스텐션 설정(`setup_ok`, `perception_python`, `datasets_root`,
  `vlm_model`, `vlm_periodic`, `autostart_config`, `quit_on_finish` 등)은 모두 그대로 동작.

## 3. 병합 후 검증

빈 폴더에서 강의 스크립트로 새로 설치(`f1d0ebc` + 패치)한 뒤 UA-DETRAC 두 영상(MVI_20011, MVI_40191)으로 실행.

| 단계 | 결과 |
| --- | --- |
| 5 실행 | 두 영상 끝까지 재생, exit 0 |
| 6 익스텐션 | 두 영상 끝까지, 25.3 FPS, 추적 결과 숫자 병합 전과 동일 |
| 7 VLM (qwen3-vl:30b-a3b-instruct, Q4_K_M) | 병합 전과 같은 답변, 7.3 / 2.3 / 2.2초 |
| 8 VLM 주기 실행 | 답변 6개 / 25개, 평균 약 3.8초, 오류 0, 사고 오탐 0, 여유 메모리 80GB 이상 |

## 4. 차선 표시: 새 방식 유지

새 scene-logic 차선 페인트 추출은 강의 영상에서 차선 표시를 거의 찾지 못합니다. 그래도 새 방식이 더 정확하므로
되돌리지 않고 그대로 씁니다.

| 영상 | 예전 방식 | 새 방식 (scene logic) |
| --- | --- | --- |
| MVI_20011 (버스 정류장) | 55조각 (캐시 기록 38개) | 0개 |
| MVI_40191 (고속도로) | 측정 안 함 | 1개 |

- 패널에 빨간 "Lane markings failed: no lane markings found"가 뜨지만, 설정 파일의 기본 차선 선과 재생·추적·VLM은 정상.
- 원인 추정: 26초짜리 영상에 차가 많아 프레임 평균으로 만든 도로 배경이 뭉개져 페인트가 보이지 않음.
- 주의: 새 코드도 `data/cache/lanes/`를 읽으므로, 예전 방식으로 만든 캐시가 남은 PC에서는 예전 차선이 계속 보입니다.
  같은 결과를 보려면 해당 캐시 파일을 지우거나 옮기세요.
- 미확인: 더 길거나 차가 적은 영상에서 새 방식이 차선을 찾는지.

## 5. 이 기기에서의 메모리 주의

DGX Spark는 CPU·GPU가 약 121GB를 공유하고, 컨테이너 cgroup 한도(약 117.7GiB)를 넘으면 데스크톱 전체가 종료됩니다.
qwen2.5vl:72b는 실제로 OOM을 냈으므로 이 기기에서는 72B 이상 VLM을 쓰지 않습니다.
