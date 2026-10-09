"""Timeline window: position slider with accident markers, transport controls, playback
speed, VLM scan and analyze-after-jump, and the list of markers to jump to."""

from typing import List, Optional, Tuple

import omni.ui as ui

SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0]
COLOR_CONTACT = 0xFF00A5FF      # orange (ABGR): annotated contact time
COLOR_VLM = 0xFF3030FF          # red: VLM accident answer
COLOR_PLAYHEAD = 0xFFFFFFFF
COLOR_BAR = 0xFF303030
MARKER_ROWS = 40


class TimelineWindow:
    def __init__(self, lead_in_s: float = 3.0, scan_step_s: float = 2.0):
        self.dual = None                  # cctv.dual_tracking extension instance
        self._dragging = False
        self._scenario_key = None
        self._markers_key = None
        self._analyze_from_frame: Optional[int] = None   # analyze the first frame shown at/after this id
        self._markers: List[Tuple[float, str, str, int]] = []   # (time s, kind, text, color)

        self.window = ui.Window("CCTV Timeline", width=900, height=520)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                self._title = ui.Label("no replay: start a scenario in 'CCTV Dual Tracking'", height=18)
                self._bar_frame = ui.Frame(build_fn=self._build_bar, height=16)
                self._pos = ui.SimpleFloatModel(0.0)
                self._slider_frame = ui.Frame(build_fn=self._build_slider, height=22)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("|<", width=40, clicked_fn=lambda: self._seek_to(0.0))
                    ui.Button("-5s", width=48, clicked_fn=lambda: self._seek_by(-5.0))
                    ui.Button("-1s", width=48, clicked_fn=lambda: self._seek_by(-1.0))
                    self._play = ui.Button("Pause", width=70, clicked_fn=self._toggle_pause)
                    ui.Button("Step", width=56, clicked_fn=self._step)
                    ui.Button("+1s", width=48, clicked_fn=lambda: self._seek_by(1.0))
                    ui.Button("+5s", width=48, clicked_fn=lambda: self._seek_by(5.0))
                    ui.Spacer(width=12)
                    ui.Label("Speed", width=40)
                    self._speed = ui.ComboBox(SPEEDS.index(1.0), *[f"{s:g}x" for s in SPEEDS], width=70)
                    self._speed.model.add_item_changed_fn(self._on_speed)
                    ui.Spacer(width=12)
                    ui.Label("Lead-in (s)", width=70)
                    self._lead = ui.SimpleFloatModel(lead_in_s)
                    ui.FloatDrag(model=self._lead, min=0.0, max=30.0, step=0.5, width=60)
                with ui.HStack(height=24, spacing=6):
                    self._analyze_after = ui.CheckBox(model=ui.SimpleBoolModel(True), width=20)
                    ui.Label("VLM-analyze the frame after each jump", width=260)
                    ui.Button("Analyze this frame now", width=180, clicked_fn=self._analyze_now)
                    self._vlm_label = ui.Label("", word_wrap=True)
                with ui.CollapsableFrame("VLM scan of the whole video (finds accident markers)", height=0):
                    with ui.VStack(spacing=4, height=0):
                        with ui.HStack(height=24, spacing=6):
                            ui.Label("Step (s)", width=60)
                            self._scan_step = ui.SimpleFloatModel(scan_step_s)
                            ui.FloatDrag(model=self._scan_step, min=0.5, max=30.0, step=0.5, width=70)
                            ui.Button("Scan", width=80, clicked_fn=self._scan)
                            ui.Button("Cancel", width=80, clicked_fn=self._cancel_scan)
                            self._scan_progress = ui.SimpleFloatModel(0.0)
                            ui.ProgressBar(model=self._scan_progress)
                        self._scan_label = ui.Label("uses the model and prompt of the VLM Console", height=18)
                with ui.HStack(height=20, spacing=6):
                    ui.Label("Markers (click Go: jump to marker minus lead-in)", width=330)
                    self._show_all = ui.CheckBox(model=ui.SimpleBoolModel(False), width=20)
                    self._show_all.model.add_value_changed_fn(lambda _m: self._force_markers())
                    ui.Label("also list non-accident VLM results")
                with ui.ScrollingFrame(height=220):
                    self._list_frame = ui.Frame(build_fn=self._build_list)

    # ------------------------------------------------------------------ pieces

    @property
    def session(self):
        return None if self.dual is None else self.dual.session

    @property
    def duration(self) -> float:
        s = self.session
        return s.duration_s if s is not None and s.total_frames else 1.0

    def _build_slider(self):
        slider = ui.FloatSlider(model=self._pos, min=0.0, max=max(self.duration, 0.1), step=0.04)
        slider.model.add_begin_edit_fn(lambda _m: setattr(self, "_dragging", True))
        slider.model.add_end_edit_fn(self._on_slider_released)

    def _build_bar(self):
        with ui.ZStack(height=14):
            ui.Rectangle(style={"background_color": COLOR_BAR})
            duration = self.duration
            for t, _kind, _text, color in self._markers:
                frac = min(max(t / duration, 0.0), 1.0)
                with ui.Placer(offset_x=ui.Percent(frac * 100.0)):
                    ui.Rectangle(width=4, height=14, style={"background_color": color},
                                 mouse_pressed_fn=lambda x, y, b, m, t=t: self._jump(t))
            s = self.session
            if s is not None:
                frac = min(max(s.last_time_s / duration, 0.0), 1.0)
                with ui.Placer(offset_x=ui.Percent(frac * 100.0)):
                    ui.Rectangle(width=2, height=14, style={"background_color": COLOR_PLAYHEAD})

    def _build_list(self):
        with ui.VStack(spacing=2, height=0):
            if not self._markers:
                ui.Label("no markers yet: the scenario has no annotated contact and the VLM reported no accident "
                         "(run a scan)", height=18, style={"color": 0xFF999999})
            for t, kind, text, color in self._markers[:MARKER_ROWS]:
                with ui.HStack(height=20, spacing=6):
                    ui.Button("Go", width=36, clicked_fn=lambda t=t: self._jump(t))
                    ui.Label(f"{t:7.2f}s", width=70)
                    ui.Label(kind, width=110, style={"color": color})
                    ui.Label(text[:140])

    # ------------------------------------------------------------------ actions

    def _seek_to(self, t: float):
        s = self.session
        if s is None:
            return
        t = min(max(t, 0.0), self.duration)
        s.seek(t)
        if self._analyze_after.model.as_bool:
            self._analyze_from_frame = int(round(t * s.fps)) + 1
        self._pos.set_value(t)

    def _seek_by(self, dt: float):
        s = self.session
        if s is not None:
            self._seek_to(s.last_time_s + dt)

    def _jump(self, t: float):
        """To a marker: a few seconds before it, so the run-up is visible."""
        self._seek_to(t - self._lead.as_float)

    def _on_slider_released(self, model):
        self._dragging = False
        self._seek_to(model.as_float)

    def _toggle_pause(self):
        s = self.session
        if s is not None:
            s.toggle_pause()

    def _step(self):
        s = self.session
        if s is None:
            return
        if s.state == s.RUNNING:
            s.toggle_pause()
        s.step()

    def _on_speed(self, model, _item):
        s = self.session
        if s is not None:
            s.speed = SPEEDS[model.get_item_value_model().as_int]

    def _analyze_now(self):
        s = self.session
        if s is None or s.last_frame is None or self.dual is None:
            self._vlm_label.text = "no frame yet"
            return
        ok = self.dual.vlm.request_now(s.last_frame, s.frame_id, s.last_time_s)
        self._vlm_label.text = (f"requested @ {s.last_time_s:.2f}s" if ok else f"VLM unavailable: {self.dual.vlm.status}")

    def _scan(self):
        s = self.session
        if s is None or self.dual is None:
            self._scan_label.text = "start a scenario first"
            return
        video = self.dual.config["video"]
        if self.dual.vlm.scan(video["path"], s.fps, self._scan_step.as_float):
            self._scan_label.text = "scanning..."
        else:
            self._scan_label.text = self.dual.vlm.scan_status or "a scan is already running"

    def _cancel_scan(self):
        if self.dual is not None:
            self.dual.vlm.cancel_scan()

    def _force_markers(self):
        self._markers_key = None

    # ------------------------------------------------------------------ refresh (extension update loop)

    def _collect_markers(self):
        markers = []
        config = self.dual.config or {}
        accident = config.get("accident") or {}
        if "contact_s" in accident:
            markers.append((float(accident["contact_s"]), "contact (annot.)",
                            str(accident.get("description", "")), COLOR_CONTACT))
        show_all = self._show_all.model.as_bool
        for r in self.dual.vlm.history:
            if r.error or not (r.accident or show_all):
                continue
            kind = ("VLM accident" if r.accident else "VLM") + (" scan" if r.prompt_name.endswith("/scan") else "")
            markers.append((r.video_time_s, kind, r.summary, COLOR_VLM if r.accident else 0xFF999999))
        self._markers = sorted(markers, key=lambda m: m[0])

    def refresh(self, dual):
        self.dual = dual
        s = self.session
        if s is None:
            self._title.text = "no replay: start a scenario in 'CCTV Dual Tracking'"
            return
        scenario_key = (id(s), s.total_frames)
        if scenario_key != self._scenario_key:          # new scenario: slider range
            self._scenario_key = scenario_key
            self._slider_frame.rebuild()
            self._speed.model.get_item_value_model().set_value(SPEEDS.index(1.0))
            self._markers_key = None

        name = (self.dual.config or {}).get("accident", {}).get("sequence", "")
        self._title.text = (f"{s.state:8} {s.last_time_s:7.2f} / {s.duration_s:.2f} s   frame {s.frame_id}/{s.total_frames}"
                            f"   {s.speed:g}x   {name}")
        self._play.text = "Play" if s.state == s.PAUSED else "Pause"
        speed_index = self._speed.model.get_item_value_model()
        if s.speed in SPEEDS and SPEEDS[speed_index.as_int] != s.speed:   # changed elsewhere
            speed_index.set_value(SPEEDS.index(s.speed))
        if not self._dragging:
            self._pos.set_value(s.last_time_s)

        markers_key = (self.dual.vlm.version, self._show_all.model.as_bool, id(self.dual.config))
        if markers_key != self._markers_key:
            self._markers_key = markers_key
            self._collect_markers()
            self._list_frame.rebuild()
        self._bar_frame.rebuild()                        # playhead

        vlm = self.dual.vlm
        if vlm.scanning or vlm.scan_status:
            self._scan_progress.set_value(vlm.scan_progress)
            self._scan_label.text = vlm.scan_status
        latest = vlm.latest
        if latest is not None:
            self._vlm_label.text = (f"VLM @{latest.video_time_s:.1f}s{' ACCIDENT' if latest.accident else ''}: "
                                    f"{(latest.error or latest.summary)[:90]}")

        # Analyze the first frame shown after a jump
        if self._analyze_from_frame is not None and s.last_frame is not None and s.frame_id >= self._analyze_from_frame:
            self._analyze_from_frame = None
            vlm.request_now(s.last_frame, s.frame_id, s.last_time_s)

    def destroy(self):
        self.window.destroy()
