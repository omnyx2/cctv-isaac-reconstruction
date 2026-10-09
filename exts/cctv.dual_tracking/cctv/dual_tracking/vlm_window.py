"""VLM Console: connection, model, timing, prompt, on-demand tests, statistics and
results of the vision-language scene analysis. Every change applies to the next
analysis; nothing here blocks the replay (perception/vlm/analyzer.py)."""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import omni.ui as ui

from .vlm_controller import VLMController, describe, now_stamp

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
THUMB_WIDTH = 360
HISTORY_ROWS = 30
LABEL_W = 110


class VLMWindow:
    def __init__(self, controller: VLMController, project_root: Path,
                 current_frame: Callable[[], Optional[tuple]]):
        """`current_frame()` -> (frame BGR, frame_id, video time s) of the replay, or None."""
        from perception.vlm import PRESETS

        self.c = controller
        self.project_root = project_root
        self.current_frame = current_frame
        self.presets = list(PRESETS) + ["custom"]
        self._picker = None
        self._shown_version = -1
        self._models = [controller.settings["model"]]
        self._thumb = ui.ByteImageProvider()

        self.window = ui.Window("CCTV VLM Console", width=620, height=900)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    self._build_connection()
                    self._build_run()
                    self._build_prompt()
                    self._build_test()
                    self._build_stats()
                    self._build_results()
        self.refresh_models()

    # ------------------------------------------------------------------ sections

    def _row(self, label: str):
        stack = ui.HStack(height=22, spacing=6)
        with stack:
            ui.Label(label, width=LABEL_W)
        return stack

    def _build_connection(self):
        with ui.CollapsableFrame("Connection", height=0):
            with ui.VStack(spacing=4, height=0):
                with self._row("Ollama URL"):
                    self._url = ui.StringField(model=ui.SimpleStringModel(self.c.settings["url"]))
                    self._url.model.add_end_edit_fn(lambda m: (self.c.set("url", m.as_string), self.refresh_models()))
                    ui.Button("Check / refresh", width=120, clicked_fn=self.refresh_models)
                with self._row("Model"):
                    self._model_frame = ui.Frame(build_fn=self._build_model_combo, height=22)
                self._conn_status = ui.Label("", height=18, word_wrap=True)

    def _build_model_combo(self):
        current = self.c.settings["model"]
        items = self._models if current in self._models else [current] + self._models
        combo = ui.ComboBox(items.index(current), *items)

        def changed(model, _item):
            self.c.set("model", items[model.get_item_value_model().as_int])
        combo.model.add_item_changed_fn(changed)

    def _build_run(self):
        s = self.c.settings
        with ui.CollapsableFrame("Run settings (live)", height=0):
            with ui.VStack(spacing=4, height=0):
                with self._row("Periodic analysis"):
                    self._periodic = ui.CheckBox(model=ui.SimpleBoolModel(self.c.periodic), width=20)
                    self._periodic.model.add_value_changed_fn(self._on_periodic)
                    ui.Label("analyze replay frames every interval", style={"color": 0xFF999999})
                with self._row("Interval (s)"):
                    m = ui.SimpleFloatModel(float(s["interval_s"]))
                    ui.FloatSlider(model=m, min=0.5, max=30.0, step=0.5)
                    m.add_value_changed_fn(lambda m: self.c.set("interval_s", round(m.as_float, 2)))
                with self._row("Image long side"):
                    m = ui.SimpleIntModel(int(s["max_side"]))
                    ui.IntSlider(model=m, min=336, max=1920, step=16)
                    m.add_value_changed_fn(lambda m: self.c.set("max_side", m.as_int))
                with self._row("Temperature"):
                    m = ui.SimpleFloatModel(float(s["temperature"]))
                    ui.FloatSlider(model=m, min=0.0, max=1.0, step=0.05)
                    m.add_value_changed_fn(lambda m: self.c.set("temperature", round(m.as_float, 2)))
                with self._row("Timeout (s)"):
                    m = ui.SimpleFloatModel(float(s["timeout_s"]))
                    ui.FloatSlider(model=m, min=5.0, max=180.0, step=5.0)
                    m.add_value_changed_fn(lambda m: self.c.set("timeout_s", round(m.as_float, 1)))
                with self._row("JSON answer"):
                    self._json = ui.CheckBox(model=ui.SimpleBoolModel(bool(s["json_format"])), width=20)
                    self._json.model.add_value_changed_fn(lambda m: self.c.set("json_format", m.as_bool))
                    ui.Label("server forces JSON (JSON prompts only)", style={"color": 0xFF999999})
                with self._row("Log to file"):
                    self._log = ui.CheckBox(model=ui.SimpleBoolModel(False), width=20)
                    self._log.model.add_value_changed_fn(self._on_log)
                    self._log_label = ui.Label("", style={"color": 0xFF999999})

    def _build_prompt(self):
        with ui.CollapsableFrame("Prompt", height=0):
            with ui.VStack(spacing=4, height=0):
                with self._row("Preset"):
                    self._preset = ui.ComboBox(self.presets.index(self.c.prompt_name)
                                               if self.c.prompt_name in self.presets else len(self.presets) - 1,
                                               *self.presets)
                    self._preset.model.add_item_changed_fn(self._on_preset)
                self._prompt = ui.StringField(model=ui.SimpleStringModel(self.c.prompt), multiline=True, height=170)
                with ui.HStack(height=26, spacing=6):
                    ui.Button("Apply prompt", clicked_fn=self._apply_prompt)
                    ui.Button("Reset to preset", clicked_fn=lambda: self._on_preset(self._preset.model, None))
                self._prompt_status = ui.Label(f"active: {self.c.prompt_name}", height=18)

    def _build_test(self):
        with ui.CollapsableFrame("Test now", height=0):
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=26, spacing=6):
                    ui.Button("Analyze current replay frame", clicked_fn=self._analyze_current)
                    ui.Button("Analyze image file...", clicked_fn=self._pick_image)
                self._test_status = ui.Label("", height=18, word_wrap=True)
                ui.ImageWithProvider(self._thumb, width=THUMB_WIDTH, height=THUMB_WIDTH * 10 // 16,
                                     fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)

    def _build_stats(self):
        with ui.CollapsableFrame("Status and timing", height=0):
            with ui.VStack(spacing=2, height=0):
                self._status = ui.Label("", height=18, word_wrap=True)
                self._stats = ui.Label("", height=36, word_wrap=True)

    def _build_results(self):
        with ui.CollapsableFrame("Results", height=0):
            with ui.VStack(spacing=4, height=0):
                self._latest = ui.Label("no result yet", height=60, word_wrap=True)
                ui.Label("Raw answer", height=16)
                self._raw = ui.StringField(model=ui.SimpleStringModel(""), multiline=True, read_only=True, height=120)
                ui.Label("History (newest first)", height=16)
                self._history_frame = ui.Frame(build_fn=self._build_history, height=0)

    def _build_history(self):
        with ui.VStack(spacing=1, height=0):
            for r in list(self.c.history)[:HISTORY_ROWS]:
                flag = "ACCIDENT " if r.accident else ""
                text = r.error or r.summary
                ui.Label(f"{r.video_time_s:7.1f}s  {r.latency_s:4.1f}s  {r.prompt_name:<14} {flag}{text}"[:170],
                         height=16, style={"color": 0xFF4040FF} if r.accident else {"color": 0xFFCCCCCC})

    # ------------------------------------------------------------------ actions

    def refresh_models(self):
        models, error = self.c.models()
        if error:
            self._conn_status.text = f"server: {error}"
            self._conn_status.style = {"color": 0xFF4040FF}
        else:
            self._models = models or [self.c.settings["model"]]
            self._conn_status.text = f"server ok: {len(models)} model(s) pulled"
            self._conn_status.style = {"color": 0xFF66CC66}
        self._model_frame.rebuild()

    def _on_periodic(self, model):
        if not self.c.set_periodic(model.as_bool):
            model.set_value(False)

    def _on_log(self, model):
        if model.as_bool:
            self.c.log_path = self.project_root / "data" / "results" / "vlm" / f"console_{now_stamp()}.jsonl"
            self._log_label.text = str(self.c.log_path.relative_to(self.project_root))
        else:
            self.c.log_path = None
            self._log_label.text = ""

    def _on_preset(self, model, _item):
        from perception.vlm import PRESETS

        name = self.presets[model.get_item_value_model().as_int]
        if name == "custom":
            return
        prompt, json_format = PRESETS[name]
        self._prompt.model.set_value(prompt)
        self._json.model.set_value(json_format)
        self.c.set_prompt(prompt, name, json_format)
        self._prompt_status.text = f"active: {name} (JSON answer {'on' if json_format else 'off'})"

    def _apply_prompt(self):
        from perception.vlm import PRESETS

        text = self._prompt.model.as_string.strip()
        if not text:
            self._prompt_status.text = "empty prompt not applied"
            return
        name = next((n for n, (p, _) in PRESETS.items() if p.strip() == text), "custom")
        self.c.set_prompt(text, name)
        if name == "custom":
            self._preset.model.get_item_value_model().set_value(len(self.presets) - 1)
        self._prompt_status.text = (f"active: {name} (JSON answer {'on' if self.c.settings['json_format'] else 'off'}; "
                                    "turn it off for free-text prompts)")

    def _analyze_current(self):
        current = self.current_frame()
        if current is None:
            self._test_status.text = "no replay frame yet: start a scenario, or analyze an image file"
            return
        frame, frame_id, t = current
        self._request(frame, frame_id, t, f"replay frame {frame_id} ({t:.1f}s)")

    def _pick_image(self):
        from omni.kit.window.filepicker import FilePickerDialog

        def apply(filename: str, dirname: str):
            self._picker.hide()
            path = Path(dirname) / filename
            frame = cv2.imread(str(path))
            if frame is None:
                self._test_status.text = f"cannot read {path}"
                return
            self._request(frame, 0, 0.0, path.name)

        self._picker = FilePickerDialog(
            "Image to analyze", apply_button_label="Analyze", click_apply_handler=apply,
            click_cancel_handler=lambda *_: self._picker.hide(),
            item_filter_options=["Images (" + ", ".join(IMAGE_EXTENSIONS) + ")", "All files (*.*)"])
        self._picker.show()

    def _request(self, frame: np.ndarray, frame_id: int, t: float, label: str):
        if self.c.request_now(frame, frame_id, t):
            self._test_status.text = f"requested: {label} (result below in a few seconds)"
        else:
            self._test_status.text = f"VLM unavailable: {self.c.status}"
        h, w = frame.shape[:2]
        size = (THUMB_WIDTH, int(round(h * THUMB_WIDTH / w)))
        rgba = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGBA)
        self._thumb.set_data_array(np.ascontiguousarray(rgba), [size[0], size[1]])

    # ------------------------------------------------------------------ refresh (called by the extension)

    def refresh(self):
        self._status.text = (f"{self.c.status} | prompt {self.c.prompt_name} | interval {self.c.settings['interval_s']}s | "
                             f"{self.c.settings['max_side']}px | periodic {'on' if self.c.periodic else 'off'}")
        self._stats.text = self.c.stats_text()
        if self.c.periodic != self._periodic.model.as_bool:
            self._periodic.model.set_value(self.c.periodic)   # changed elsewhere (scenario config)
        if self._shown_version != self.c.version:
            self._shown_version = self.c.version
            latest = self.c.latest
            if latest is not None:
                self._latest.text = describe(latest)
                self._latest.style = {"color": 0xFF4040FF} if latest.accident else {}
                self._raw.model.set_value(latest.raw or latest.error or "")
            self._history_frame.rebuild()

    @property
    def visible(self) -> bool:
        return self.window.visible

    def show(self):
        self.window.visible = True

    def destroy(self):
        if self._picker is not None:
            self._picker.destroy()
            self._picker = None
        self.window.destroy()
