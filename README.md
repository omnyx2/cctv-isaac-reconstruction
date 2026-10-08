# CCTV Dual-Tracking

원본 CCTV 영상에서 차량·보행자를 추적하고, 그 움직임을 Isaac Sim 3D 시뮬레이션에 실시간으로 재현한다.
시뮬레이션의 차량은 자전거 모델로 "운전"되므로, 실제 차가 할 수 없는 움직임(옆미끄러짐, 제자리 회전, 순간이동)이 없다.
현실에 없는 충돌(차량끼리 겹침)도 만들지 않는다.

```
원본 CCTV ─▶ YOLOv8 + ByteTrack ─▶ 흔들림 보정 ─▶ 지면 투영 ─▶ 동역학 필터 ─▶ 물리적 일관성 ─▶ 자전거 모델 운전 ─▶ Isaac Sim
            (인식 프로세스)                       (월드 모델, world/)                                    (simulation/)
```

## 실행

Isaac Sim 5.1 (`C:\isaacsim`) 번들 Python으로 실행한다. 반드시 `python.bat`을 쓴다.

```
C:\isaacsim\python.bat main.py --config config/detrac_MVI_40191.yaml
```

| 옵션 | 설명 |
|---|---|
| `--config` | 시퀀스 설정 (기본값 `config/detrac_MVI_20011.yaml`) |
| `--profile` | 종료할 때 단계별 프레임 시간 출력 |
| `--save-dir DIR --save-every N` | 실제 \| 시뮬레이션을 나란히 붙인 스냅샷 저장 |

창에서 `Q` 종료, `Space` 일시정지, `R` 처음부터 다시 재생.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 코드 구조, 계층·의존 규칙, 프로세스 모델, 데이터 흐름, 주요 클래스, 작업 방법 |
| [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md) | 적용 알고리즘과 기술 — 어디에, 무엇을, 어떤 문제를 해결했는지와 측정 수치 |
| [docs/DATA_POLICY.md](docs/DATA_POLICY.md) | **원본(raw) CCTV만 사용한다**는 규칙과 보유 데이터 판정 |
| [docs/SESSION_LOG_2026-10-08.md](docs/SESSION_LOG_2026-10-08.md) | 작업 경위 (시간순) |

## 현재 성능 (UA-DETRAC)

| 시퀀스 | 추적 MOTA | 실시간 | 옆미끄러짐 평균 (운전) | 차량 겹침 (출력) |
|---|---|---|---|---|
| MVI_20011 (정류장, 주차열, 보행자) | 0.645 | 24.4 FPS | 0.142 m/s | 0 |
| MVI_40191 (고속화도로, 양방향) | 0.874 | 24.9 FPS | 0.036 m/s | 0 |

## 위험 장면 탐색 (오프라인)

UA-DETRAC에는 사고 라벨이 없다. 그래서 추적 결과로 아찔한 순간(추돌 위험 TTC, 근접 추종, 급감속, 급차선변경)을 찾는다.
카메라는 교통 자체로 자동 추정하고, 초점거리를 몰라도 정확한 비율 지표만 쓴다.
60개 시퀀스에서 상충 기준을 넘은 장면은 12건, 근접 후보는 77건이었다. 직접 확인해 보니 뚜렷한 근접 사고는 거의 없었다.

```
C:\isaacsim\python.bat tools/cache_detections.py        # 60개 시퀀스 인식 캐시 (약 50분)
C:\isaacsim\kit\python\python.exe tools/mine_near_misses.py --top 20
```
방법과 한계는 TECHNICAL_NOTES 12장, 결과는 `data/results/near_miss/REPORT.md`에 있다.

## 디렉터리

```
main.py        앱 진입점
perception/    영상 → 추적 결과 (YOLO, 흔들림 보정, 인식 프로세스)
world/         월드 모델 (투영, 필터, 일관성, 차량 운전) — 순수 numpy
simulation/    Isaac Sim 렌더링 (카메라, 차량, 보행자)
analysis/      오프라인 분석: 자동 도로 기하, 위험 장면 지표
tools/         캘리브레이션, 캐시, 평가, 위험 장면 탐색 도구
config/        시퀀스별 설정, TEMPLATE.yaml
docs/          문서
data/          데이터셋·영상·결과 (git 제외)
```
