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

### Isaac Sim 확장 (영상을 골라서 실행)

```
C:\isaacsim\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking --enable cctv.timeline
```
**CCTV Dual Tracking** 창에서 시나리오나 영상(파일 또는 프레임 폴더)을 골라 Start를 누르면 뷰포트에서 재구성이 돈다
([exts/cctv.dual_tracking/docs/README.md](exts/cctv.dual_tracking/docs/README.md)).
**CCTV VLM Console**에서 VLM 모델·주기·프롬프트를 바꾸고 결과와 처리 시간을 본다.
**CCTV Timeline**에서 사고 마커(주석 접촉 시각, VLM 판정)를 눌러 그 몇 초 앞으로 이동하고, 슬로모션·한 프레임씩 넘기기, 영상 전체 VLM 스캔을 한다
([exts/cctv.timeline/docs/README.md](exts/cctv.timeline/docs/README.md)).

### VLM 장면 분석 (선택)

YOLO와 별개로 비전-언어 모델이 장면을 설명한다(사고·연기·도로 위 사람 등). Ollama를 설치하고 `ollama pull qwen2.5vl:7b`를 한 뒤
설정에서 `vlm.enabled: true`로 켠다. 오프라인 분석: `C:\isaacsim\kit\python\python.exe tools/vlm_analyze.py --config <설정>`.

## 설치 — Isaac Sim이 있는 다른 컴퓨터 (예: DGX Spark)

이 저장소에는 **코드만** 있다. 데이터셋, 모델 가중치(`*.pt`), 캐시(`data/`), 문서와 사진(`docs/`)은 들어 있지 않다.

1. **코드를 받는다.** 둘 중 하나.
   - 터미널에서 `git clone https://github.com/omnyx2/cctv-isaac-reconstruction ~/CCTV_Project`
     - 비공개 저장소라 그 컴퓨터에서 `gh auth login`을 한 번 해 둔다.
   - Isaac Sim의 Window → Extensions → ☰ → Settings → Extension Search Paths에
     `git+https://github.com/omnyx2/cctv-isaac-reconstruction.git?branch=master&dir=exts`를 넣는다.
     - 이 방법은 Kit에 git 확장 경로 지원(`omni.kit.extpath.git`)과 git 인증이 있어야 한다.
     - 안 되면 위의 `git clone` 후 `~/CCTV_Project/exts`를 넣는다.
2. **확장을 켠다.** Extensions에서 `CCTV`를 검색해 **CCTV Timeline**을 켜고 Autoload를 체크한다. Dual Tracking은 같이 켜진다.
3. **CCTV Setup 창에서 설정한다.** 창은 처음에 저절로 뜬다. 위에서부터 차례로 한다.
   - 데이터셋 폴더를 지정한다.
   - **Install packages**: Isaac Python에 패키지를 설치한다.
   - **Import**: 데이터 전송 폴더를 가져온다. 전송 폴더는 원래 PC에서 `tools/pack_for_spark.py`로 만든다(모델, 캐시, 데이터셋).
   - **Check environment**: 환경을 점검한다.
   - **Create GPU venv**: 필요할 때만. Isaac 내장 torch가 GPU를 못 쓸 때(예: GB10 = sm_121) PyTorch cu130 venv를 만든다.
   - **Update code**: 이후 코드 업데이트는 이 버튼(git pull)으로 한다.

터미널만으로 하려면 `tools/setup_spark.sh`를 쓴다. 이 스크립트는 Isaac Sim 소스 빌드부터 시작한다.

## 현재 성능 (UA-DETRAC)

| 시퀀스 | 추적 MOTA | 실시간 | 옆미끄러짐 평균 (운전) | 차량 겹침 (출력) |
|---|---|---|---|---|
| MVI_20011 (정류장, 주차열, 보행자) | 0.645 | 24.4 FPS | 0.085 m/s | 0 |
| MVI_40191 (고속화도로, 양방향) | 0.874 | 24.9 FPS | 0.033 m/s | 0 |

## 위험 장면 탐색 (오프라인)

UA-DETRAC에는 사고 라벨이 없다. 그래서 추적 결과로 아찔한 순간(추돌 위험 TTC, 근접 추종, 급감속, 급차선변경)을 찾는다.
카메라는 교통 자체로 자동 추정하고, 초점거리를 몰라도 정확한 비율 지표만 쓴다.
60개 시퀀스에서 상충 기준을 넘은 장면은 12건, 근접 후보는 77건이었다. 직접 확인해 보니 뚜렷한 근접 사고는 거의 없었다.

```
C:\isaacsim\python.bat tools/cache_detections.py        # 60개 시퀀스 인식 캐시 (약 50분)
C:\isaacsim\kit\python\python.exe tools/mine_near_misses.py --top 20
```
방법과 한계는 TECHNICAL_NOTES 12장, 결과는 `data/results/near_miss/REPORT.md`에 있다.

## 실제 사고 검증 (TUM Traffic Accid3nD)

실제 고속도로 사고 영상(고정 카메라 원본)으로 검증했다. 12개 시퀀스 중 접촉 장면이 보이는 3건을 **하나도 잡지 못했다**.
사고 순간의 비정상 움직임을 추적 오류 필터가 걸러 내기 때문이다. 새로 넣은 **차로 정지 차량** 탐지는 s01의 원인 차량을 잡았다.
사고 장면은 `D:\datasets\TUMTraf-Accid3nD\accident_scenes\`에 따로 분리했다. 장면별 실행 설정은 `config/tum_*.yaml`이다:
```
C:\isaacsim\python.bat main.py --config config/tum_r09_s01_south50.yaml
```
자세한 내용은 TECHNICAL_NOTES 13~15장에 있다.

## 디렉터리

```
main.py        앱 진입점
perception/    영상 → 추적 결과 (YOLO, 흔들림 보정, 인식 프로세스), vlm/ (VLM 장면 분석)
world/         월드 모델 (투영, 필터, 일관성, 차량 운전) — 순수 numpy
simulation/    Isaac Sim 렌더링 (장면, 카메라, 차량, 보행자)
exts/          Isaac Sim 확장 cctv.dual_tracking (영상 선택 → 시뮬레이션)
analysis/      오프라인 분석: 자동 도로 기하, 위험 장면 지표, TUM 사고 데이터
tools/         캘리브레이션, 캐시, 평가, 위험 장면 탐색 도구
config/        시퀀스별 설정, TEMPLATE.yaml
docs/          문서 (로컬 전용, 저장소에 없음)
data/          데이터셋·영상·결과 (git 제외)
```
