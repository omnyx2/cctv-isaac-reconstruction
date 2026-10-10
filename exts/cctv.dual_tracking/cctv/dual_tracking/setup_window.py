"""The "CCTV Setup" window: get the project running in an Isaac Sim that is installed
already (DGX Spark or any other machine), without a terminal. Engine: setup_env.py."""

import platform
import sys
from pathlib import Path
from typing import List, Optional

import omni.ui as ui

from . import setup_env as env

OK, BAD, RUN, DIM = 0xFF66DD66, 0xFF5555EE, 0xFF66CCFF, 0xFF999999
LABEL_W = 120


class SetupWindow:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.job: Optional[env.Job] = None
        self._shown_lines = -1
        self._picker = None
        self.window = ui.Window("CCTV Setup", width=720, height=760, visible=False)
        with self.window.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=8, height=0):
                    self._build()

    # ------------------------------------------------------------------ layout

    def _build(self):
        isaac_version = "?"
        try:
            import omni.kit.app
            isaac_version = omni.kit.app.get_app().get_app_version()
        except Exception:
            pass
        ui.Label(f"{platform.system()} {platform.machine()} | Kit {isaac_version} | Python {sys.version.split()[0]}",
                 height=18, style={"color": DIM})
        ui.Label(f"Project: {self.project_root}", height=18, style={"color": DIM}, word_wrap=True)

        with ui.CollapsableFrame("1  Datasets folder  (${DATASETS} in the configs)", height=0):
            with ui.HStack(height=24, spacing=6):
                ui.Label("Folder", width=LABEL_W)
                from cctv_utils.paths import datasets_root
                self._datasets = ui.StringField(model=ui.SimpleStringModel(str(env.get(env.KEY_DATASETS)
                                                                               or datasets_root())))
                ui.Button("Browse...", width=80, clicked_fn=lambda: self._pick_folder(self._datasets))
                ui.Button("Apply", width=60, clicked_fn=self._apply_datasets)

        with ui.CollapsableFrame("2  Packages in Isaac Sim's Python", height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label("numpy 1.26, OpenCV, lap, ultralytics (with --no-deps: it must not pull a second torch)",
                         word_wrap=True, height=0, style={"color": DIM})
                ui.Button("Install packages", height=28, clicked_fn=self._install)

        with ui.CollapsableFrame("3  Python for YOLO (GPU)", height=0):
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=24, spacing=6):
                    ui.Label("In use", width=LABEL_W)
                    self._python_label = ui.Label("", word_wrap=True)
                with ui.HStack(height=24, spacing=6):
                    ui.Label("GPU venv", width=LABEL_W)
                    self._venv = ui.StringField(model=ui.SimpleStringModel(str(env.default_venv())))
                with ui.HStack(height=28, spacing=6):
                    ui.Button("Check environment", clicked_fn=self._check)
                    ui.Button("Create GPU venv (PyTorch cu130)", clicked_fn=self._make_venv)
                    ui.Button("Use Isaac Sim's", width=120, clicked_fn=self._use_isaac_python)
                ui.Label("If the check says the GPU is not covered (e.g. DGX Spark GB10 = sm_121), create the venv: "
                         "YOLO then runs with it; the replay itself stays in Isaac Sim.",
                         word_wrap=True, height=0, style={"color": DIM})

        with ui.CollapsableFrame("4  Models, caches, datasets from the transfer folder", height=0):
            with ui.HStack(height=24, spacing=6):
                ui.Label("Folder", width=LABEL_W)
                self._transfer = ui.StringField(model=ui.SimpleStringModel(str(Path.home() / "cctv_transfer")))
                ui.Button("Browse...", width=80, clicked_fn=lambda: self._pick_folder(self._transfer))
                ui.Button("Import", width=60, clicked_fn=self._import)

        with ui.CollapsableFrame("5  VLM (optional): Ollama", height=0):
            with ui.HStack(height=28, spacing=6):
                self._ollama_label = ui.Label("", word_wrap=True)
                ui.Button(f"Pull {env.VLM_MODEL}", width=170, clicked_fn=self._pull)

        with ui.HStack(height=24, spacing=6):
            self._layout = ui.CheckBox(width=20)
            self._layout.model.set_value(bool(env.get(env.KEY_LAYOUT, False)))
            self._layout.model.add_value_changed_fn(lambda m: env.put(env.KEY_LAYOUT, m.get_value_as_bool()))
            ui.Label("At startup: docked layout and the default scenario previewed")

        ui.Spacer(height=4)
        self._job_label = ui.Label("", height=20, word_wrap=True)
        self._checks = ui.VStack(spacing=2, height=0)
        with ui.CollapsableFrame("Output", height=0, collapsed=False):
            with ui.ScrollingFrame(height=220):
                self._log = ui.Label("", word_wrap=True, alignment=ui.Alignment.LEFT_TOP,
                                     style={"font_size": 13, "color": 0xFFCCCCCC})
        self.refresh_static()

    # ------------------------------------------------------------------ actions

    def _start(self, job: env.Job):
        if self.job is not None and not self.job.finished:
            self._job_label.text = f"busy: {self.job.title}"
            return
        self.job = job
        self._shown_lines = -1

    def _apply_datasets(self):
        env.put(env.KEY_DATASETS, self._datasets.model.get_value_as_string().strip())
        env.apply_to_environment()
        self.refresh_static()

    def _install(self):
        self._start(env.install_packages(self.project_root))

    def _check(self):
        self._start(env.check_environment(self.project_root))

    def _make_venv(self):
        self._start(env.create_gpu_venv(self.project_root, Path(self._venv.model.get_value_as_string().strip())))

    def _use_isaac_python(self):
        env.put(env.KEY_PERCEPTION, "")
        env.apply_to_environment()
        self.refresh_static()

    def _import(self):
        self._apply_datasets()
        from cctv_utils.paths import datasets_root
        self._start(env.import_transfer(self.project_root, Path(self._transfer.model.get_value_as_string().strip()),
                                        datasets_root()))

    def _pull(self):
        if env.ollama_models() is None:
            self._job_label.text = ("Ollama is not running. Install it in a terminal: "
                                    "curl -fsSL https://ollama.com/install.sh | sh")
            return
        self._start(env.pull_vlm_model(self.project_root))

    def _pick_folder(self, field):
        from omni.kit.window.filepicker import FilePickerDialog

        def apply(filename: str, dirname: str):
            field.model.set_value(str(Path(dirname) / filename) if filename else dirname)
            self._picker.hide()

        self._picker = FilePickerDialog("Select a folder", apply_button_label="Select",
                                        click_apply_handler=apply, click_cancel_handler=lambda *_: self._picker.hide())
        self._picker.show()

    # ------------------------------------------------------------------ display

    def refresh_static(self):
        python = env.get(env.KEY_PERCEPTION)
        self._python_label.text = python or f"Isaac Sim's ({env.isaac_python()})"
        models = env.ollama_models()
        self._ollama_label.text = ("not running (optional)" if models is None else
                                   f"running; {env.VLM_MODEL} {'present' if env.VLM_MODEL in models else 'missing'}")

    def poll(self):
        """Called on Kit's update while the window is visible."""
        job = self.job
        if job is None or len(job.lines) == self._shown_lines and not job.finished:
            return
        if len(job.lines) != self._shown_lines:
            self._shown_lines = len(job.lines)
            self._log.text = "\n".join(job.lines[-120:])
            self._show_checks(job.lines)
        color = {env.Job.RUNNING: RUN, env.Job.DONE: OK}.get(job.state, BAD)
        self._job_label.text = f"{job.title}: {job.state}"
        self._job_label.style = {"color": color}
        if job.finished:
            if "check" in job.title.lower():
                env.put(env.KEY_SETUP_OK, job.state == env.Job.DONE)
            self.refresh_static()
            self.job = None             # the output stays on screen

    def _show_checks(self, lines: List[str]):
        """check_env.py's [ok] / [!!] lines, coloured."""
        rows = [line for line in lines if line.startswith("[ok]") or line.startswith("[!!]") or line.startswith("      ->")]
        self._checks.clear()
        with self._checks:
            for line in rows[-24:]:
                good = line.startswith("[ok]")
                ui.Label(line, height=16, style={"color": OK if good else (BAD if line.startswith("[!!]") else DIM),
                                                 "font_size": 13})

    @property
    def visible(self) -> bool:
        return self.window.visible

    def show(self):
        self.window.visible = True
        self.refresh_static()

    def destroy(self):
        if self.job is not None:
            self.job.cancel()
        if self._picker is not None:
            self._picker.destroy()
        self.window.destroy()
