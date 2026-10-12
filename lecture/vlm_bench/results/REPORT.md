# VLM benchmark on DGX Spark (GB10, 128 GB unified memory)

Generated 2026-10-11 02:27 by `vlm_bench.py --report`. Ollama, temperature 0, one request at a time, num_ctx 8192, Isaac Sim not running. 32 CCTV frames, 18 crash photos, 8 look-alike photos.

## Speed and memory

| Model | Memory (GB) | Load (s) | Scene JSON answer (s) | Image+prompt eval (s) | Answer tokens/s | Errors |
|---|---|---|---|---|---|---|
| gemma4:e2b | 1.8 | 6.5 | 1.3 | 0.21 | 157.2 | 0 |
| qwen2.5vl:7b | 5.9 | 3.3 | 3.2 | 0.44 | 59.3 | 0 |
| qwen3-vl:30b-a3b-instruct-q8_0 | 33.5 | 8.9 | 3.4 | 0.78 | 71.2 | 0 |
| qwen3-vl:30b-a3b-instruct | 19.8 | 5.7 | 2.5 | 0.72 | 112.5 | 0 |

## Task quality

| Model | T1 JSON valid | T1 accident right | T1 emergency right | T2 crashes found (of 18) | T2 false alarms: CCTV (of 32) | T2 false alarms: look-alikes (of 8) | T3 count in range | T3 mean miss | T4 attributes right | T5 Korean |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:e2b | 100% | 78% | 91% | 5 | 0 | 0 | 47% | 1.19 | 92% | 100% |
| qwen2.5vl:7b | 100% | 93% | 93% | 16 | 0 | 0 | 75% | 0.28 | 85% | 100% |
| qwen3-vl:30b-a3b-instruct-q8_0 | 100% | 98% | 91% | 17 | 0 | 0 | 69% | 0.38 | 88% | 100% |
| qwen3-vl:30b-a3b-instruct | 100% | 100% | 91% | 17 | 0 | 0 | 69% | 0.31 | 93% | 100% |

## Not measured

- qwen3-vl:32b-instruct: the whole desktop container was killed by an OOM during this run (2026-10-10 22:47, while another model was downloading); not rerun
- qwen2.5vl:72b: OOM again (2026-10-10 ~23:54, ~5 min after measurement started, est. 60GB); mem_guard did not catch it; not rerun
- llama3.2-vision:90b: not run; user decision 2026-10-11: models this large (72b and up) are not run on this DGX Spark after two OOMs
- llama4:scout: not run; same reason (larger than 72b)

## Korean answers (T5: s20011_0001, c39851, acc6, neg3)

- **gemma4:e2b**
  - 제공해주신 이미지는 **도시의 교통 상황을 보여주는 CCTV 화면**으로 보입니다.  이 장면은 **다수의 차량들이 도로를 따라 이동하고 있으며, 보행자와 자전거 이용자들도 함께 도로를 이용하는 복잡한 도시 교통 환경**을 포착하고 있습니다.
  - 제공해주신 이미지는 **야간 도로의 교통 상황을 포착한 CCTV 영상**으로 보입니다.  도로 위에는 여러 대의 차량들이 주행하고 있으며, 노면과 차선 표시, 그리고 주변에 설치된 광고판들이 보입니다.
  - 이 CCTV 장면은 도로 위에서 두 대의 차량과 두 사람이 포착된 모습을 담고 있습니다. 한쪽에는 은색 차량과 분홍색 차량이 나란히 있으며, 그 주변으로 두 사람이 서 있거나 걷고 있는 모습이 보입니다.
  - 이 CCTV 장면은 **고속도로에서 차량들이 이동하고 있는 모습을 포착하고 있습니다.** 도로 양쪽으로는 나무들이 보이며, 여러 대의 트럭과 승용차들이 차선을 따라 주행하고 있습니다.
- **qwen2.5vl:7b**
  - 이 CCTV 장면은 도시의 버스 정류장과 주변 도로를 보여줍니다. 버스 정류장에는 몇몇 사람들이 기다리고 있으며, 도로에는 여러 대의 차량이 정차하거나 이동하고 있습니다.
  - 이 CCTV 장면은 야간에 촬영된 도로를 보여줍니다. 도로 양쪽에는 버스 정류장이 있고, 중앙에는 노란색 선이 그려져 있습니다. 도로 위에는 차량들이 정지하거나 움직이고 있으며, 일부 차량의 라이트가 밝게 빛나고 있습니다.
  - 1. 두 대의 차량이 교차로에서 충돌한 것으로 보입니다. 2. 차량 주변에는 두 명의 인물이 서 있으며, 한 명은 흰색 코트를 입고 있고 다른 한 명은 검은색 정장을 입고 있습니다.
  - 1. 고속도로의 한쪽 차선이 차량으로 가득 차 있으며, 다른 차선은 차량이 적습니다. 2. 도로 양쪽에는 나무가 자라고 있으며, 하늘은 흐린 날씨입니다.
- **qwen3-vl:30b-a3b-instruct-q8_0**
  - 도로 위에는 여러 대의 차량이 오가는 중이며, 좌측에는 버스 정류장에서 사람들이 대기하고 있다. 정류장 옆에는 자전거를 타고 있는 사람과 보행자들이 보이며, 도로 중앙에는 차량들이 주행하고 있다.
  - 밤에 촬영된 도로의 CCTV 영상이다. 도로 중앙에는 분리대가 있고, 양쪽 차선에는 흰색 자동차와 노란 버스 등 여러 차량이 주행하고 있다.
  - 도로 위 횡단보도에서 두 대의 자동차가 충돌한 사고 현장이다. 한 여성과 한 남성이 차량 옆에 서서 사고 상황을 확인하고 있다.
  - 비가 오는 날씨에 고속도로 한쪽 차선이 막히고 있으며, 대형 트럭과 승용차들이 긴 줄을 이루고 있다. 반대편 차선은 여전히 통행이 가능하지만, 전체적으로 교통 상황이 혼잡한 모습이다.
- **qwen3-vl:30b-a3b-instruct**
  - 이 장면은 도시의 다중 차선 도로를 위에서 내려다본 것으로, 중앙 분리대를 사이에 두고 양방향으로 차량이 흐르고 있다. 좌측에는 버스 정류장이 있고, 사람들이 승차를 기다리고 있으며, 일부는 자전거를 타고 있다.
  - 밤에 촬영된 도로의 CCTV 영상이다. 도로 중앙선을 따라 차량들이 주행하고 있으며, 양쪽에는 버스 정류장과 사람들이 보인다.
  - 교차로의 횡단보도에서 회색 스테이션 왜건과 핑크색 세단이 충돌한 사고 현장이다. 두 차량 사이에 서 있는 여성과 남성이 사고를 확인하고 있는 모습이다.
  - 이 영상은 흐린 날씨 속 고속도로의 교통 상황을 보여주고 있다. 왼쪽 차선은 원활하게 통행하고 있지만, 오른쪽 차선은 트럭과 자동차로 인해 심각한 정체를 겪고 있다.
