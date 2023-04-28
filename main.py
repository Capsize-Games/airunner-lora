import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
from airunner.extensions import BaseExtension
from aihandler.qtvar import Var, StringVar, FloatVar, BooleanVar
import torch
from collections import defaultdict
from diffusers.loaders import LoraLoaderMixin
from safetensors.torch import load_file

class LoraVar(Var):
    my_signal = pyqtSignal(str, float, bool)

    def __init__(self, app=None, name="", scale=1.0, enabled=False):
        self.name = StringVar("")
        self.scale = FloatVar(1.0)
        self.enabled = BooleanVar(False)

        super().__init__(app, None)

        self.name.set(name, skip_save=True)
        self.scale.set(scale, skip_save=True)
        self.enabled.set(enabled, skip_save=True)

        self.name.my_signal.connect(self.emit)
        self.scale.my_signal.connect(self.emit)
        self.enabled.my_signal.connect(self.emit)

    def emit(self):
        name = self.name.get() if self.name.get() is not None else ""
        scale = self.scale.get() if self.scale.get() is not None else 1.0
        enabled = self.enabled.get() if self.enabled.get() is not None else False
        self.my_signal.emit(name, scale, enabled)


class Settings:
    def __init__(self, app):
        self.available_loras = []


class Extension(BaseExtension):
    extension_directory = "airunner-lora"
    lora_loaded = False

    def __init__(self, settings_manager=None):
        super().__init__(settings_manager)
        self._available_loras = {}
        if "available_loras" not in self.settings_manager.settings.__dict__:
            self.settings_manager.settings.available_loras = {}

    def available_loras(self, tab_name):
        if tab_name not in self._available_loras:
            self._available_loras[tab_name] = []
            loras_path = os.path.join(self.model_base_path, "lora")
            possible_line_endings = ["ckpt", "safetensors", "bin"]
            self.settings_manager.enable_save()

            for f in os.listdir(loras_path):
                if f.split(".")[-1] in possible_line_endings:
                    name = f.split(".")[0]
                    scale = 100.0
                    enabled = True
                    # check if we have scale in self.settings_manager.settings.available_loras[tab_name]
                    if tab_name in self.settings_manager.settings.available_loras:
                        loras = self.settings_manager.settings.available_loras[tab_name]
                        for lora in loras:
                            if lora["name"] == name:
                                scale = lora["scale"]
                                enabled = lora["enabled"]
                                break
                    self._available_loras[tab_name].append({
                        "name": name,
                        "scale": scale,
                        "enabled": enabled
                    })
            self.settings_manager.settings.available_loras[tab_name] = self._available_loras[tab_name]
            self.settings_manager.save_settings()
        return self.settings_manager.settings.available_loras[tab_name]

    def generator_tab_injection(self, tab, tab_name=None):
        container = QWidget()
        container.setLayout(QVBoxLayout())
        for lora in self.available_loras(tab_name):
            lora_widget = self.load_template("lora")
            lora_widget.enabledCheckbox.setText(lora["name"])
            scale = lora["scale"]
            enabled = lora["enabled"]
            lora_widget.scaleSlider.setValue(int(scale))
            lora_widget.scaleSpinBox.setValue(scale / 100)
            lora_widget.enabledCheckbox.setChecked(enabled)
            container.layout().addWidget(lora_widget)
            lora_widget.scaleSlider.valueChanged.connect(
                lambda value, _lora_widget=lora_widget, _lora=lora, _tab_name=tab_name:
                    self.handle_lora_slider(_lora, _lora_widget, value, _tab_name))
            lora_widget.scaleSpinBox.valueChanged.connect(
                lambda value, _lora_widget=lora_widget, _lora=lora, _tab_name=tab_name:
                    self.handle_lora_spinbox(_lora, _lora_widget, value, _tab_name))
            lora_widget.enabledCheckbox.stateChanged.connect(
                lambda value, _lora=lora, _tab_name=tab_name:
                    self.toggle_lora(lora, value, _tab_name))
        # add a vertical spacer to the end of the container
        container.layout().addStretch()
        tab.PromptTabsSection.addTab(container, "LoRA")

    def toggle_lora(self, lora, value, tab_name):
        for n in range(len(self.available_loras(tab_name))):
            if self.settings_manager.settings.available_loras[tab_name][n]["name"] == lora["name"]:
                self.settings_manager.settings.available_loras[tab_name][n]["enabled"] = value == 2
        self.settings_manager.save_settings()

    def handle_lora_slider(self, lora, lora_widget, value, tab_name):
        for n in range(len(self.available_loras(tab_name))):
            if self.settings_manager.settings.available_loras[tab_name][n]["name"] == lora["name"]:
                self.settings_manager.settings.available_loras[tab_name][n]["scale"] = value / 100
        lora_widget.scaleSpinBox.setValue(lora["scale"])
        self.settings_manager.save_settings()

    def handle_lora_spinbox(self, lora, lora_widget, value, tab_name):
        for n in range(len(self.available_loras(tab_name))):
            if self.settings_manager.settings.available_loras[tab_name][n]["name"] == lora["name"]:
                self.settings_manager.settings.available_loras[tab_name][n]["scale"] = value * 100
        lora_widget.scaleSlider.setValue(int(lora["scale"]))
        self.settings_manager.save_settings()

    def generate_data_injection(self, data):
        for lora in self.available_loras(data["action"]):
            if lora["enabled"]:
                data["options"]["lora"] = [(lora.name.get(), lora.scale.get())]
        return data

    def call_pipe(self, options, model_base_path, pipe, **kwargs):
        if not self.lora_loaded:
            for lora in options["lora"]:
                path = os.path.join(model_base_path, "lora")
                filepath = None
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.startswith(lora[0]):
                            filepath = os.path.join(root, file)
                            break
                try:
                    pipe = self.load_lora(pipe, filepath)
                except RuntimeError as e:
                    continue
            self.lora_loaded = True
        return pipe

    # https://github.com/huggingface/diffusers/issues/3064
    def load_lora(self, pipeline, checkpoint_path, multiplier=1.0, device="cuda", dtype=torch.float16):
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        # load LoRA weight from .safetensors
        state_dict = load_file(checkpoint_path, device=device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(dtype)
            weight_down = elems['lora_down.weight'].to(dtype)
            try:
                alpha = elems['alpha']
                if alpha:
                    alpha = alpha.item() / weight_up.shape[1]
                else:
                    alpha = 1.0
            except KeyError:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2),
                                                                        weight_down.squeeze(3).squeeze(2)).unsqueeze(
                    2).unsqueeze(3)
            else:
                # print the shapes of weight_up and weight_down:
                # print(weight_up.shape, weight_down.shape)
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

        return pipeline
