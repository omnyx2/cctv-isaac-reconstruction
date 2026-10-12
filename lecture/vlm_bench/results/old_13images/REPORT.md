# VLM benchmark on DGX Spark (GB10, 128 GB unified memory)

Generated 2026-10-10 22:22 by `vlm_bench.py --report`. Ollama, temperature 0, one request at a time, Isaac Sim not running.

## Speed and memory

| Model | Memory (GB) | Load (s) | Scene JSON answer (s) | Image+prompt eval (s) | Answer tokens/s | Errors |
|---|---|---|---|---|---|---|
| gemma4:e2b | 2.6 | 6.4 | 1.9 | 0.72 | 154.2 | 0 |
| qwen2.5vl:7b | 13.5 | 4.1 | 3.4 | 0.44 | 56.8 | 0 |
| qwen3-vl:30b-a3b-instruct | 46.0 | 10.4 | 2.7 | 0.60 | 112.1 | 0 |

## Task quality

| Model | T1 JSON valid | T1 accident right | T1 emergency right | T2 crashes found (of 5) | T2 false alarms (of 8) | T3 count in range | T3 mean miss | T4 attributes right | T5 Korean |
|---|---|---|---|---|---|---|---|---|---|
| gemma4:e2b | 100% | 85% | 100% | 3 | 0 | 50% | 1.12 | 91% | 100% |
| qwen2.5vl:7b | 100% | 100% | 100% | 5 | 0 | 50% | 0.62 | 81% | 100% |
| qwen3-vl:30b-a3b-instruct | 100% | 100% | 100% | 5 | 0 | 62% | 0.38 | 91% | 100% |

## Korean answers (T5: bus stop frame / overturned car photo)

- **gemma4:e2b**
  - 제공해주신 이미지는 **도시의 교통 상황을 보여주는 CCTV 화면**으로 보입니다.  이 장면은 **다수의 차량들이 도로를 따라 이동하고 있으며, 보행자와 자전거 이용자들도 함께 도로를 이용하는 복잡한 도시 교통 환경**을 포착하고 있습니다.
  - 이 CCTV 영상은 도로에서 발생한 교통사고 현장을 포착하고 있습니다. 여러 차량과 경찰 인력이 현장에 배치되어 사고 수습 및 통제 작업이 진행 중인 것으로 보입니다.
- **qwen2.5vl:7b**
  - 이 CCTV 장면은 도시의 버스 정류장과 주변 도로를 보여줍니다. 버스 정류장에는 몇몇 사람들이 기다리고 있으며, 도로에는 여러 대의 차량이 정차하거나 이동하고 있습니다.
  - 이 장면은 교차로에서 교통사고가 발생한 모습을 보여줍니다. 사고로 인해 차량이 перевернулась, 그리고 주변에는 구조대와 경찰이 현장을 조사하고 있습니다.
- **qwen3-vl:30b-a3b-instruct**
  - 이미지는 도시의 다중 차선 도로를 위에서 내려다본 전경을 보여주고 있다. 좌측에는 버스 정류장에서 승객들이 대기하고 있으며, 주변에는 자동차와 자전거가 정차해 있다.
  - 이 장면은 도로 교차로에서 발생한 교통사고 현장을 보여주고 있다. 검은색 승용차가 뒤집힌 채 도로 위에 정지해 있고, 주변에는 경찰과 구급대원들이 사고를 처리하고 있다.
