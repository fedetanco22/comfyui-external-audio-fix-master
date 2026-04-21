import io
import os
import tempfile

try:
    from folder_paths import get_annotated_filepath
except ImportError:
    def get_annotated_filepath(f):
        return f


class ComfyUIDeployExternalAudio:
    """Fixed version: robust multi-backend audio loading from URLs."""
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"
    CATEGORY = "🔗ComfyDeploy"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_id": ("STRING", {"multiline": False, "default": "input_audio"}),
                "audio_file": ("STRING", {"default": ""}),
            },
            "optional": {
                "default_value": ("AUDIO",),
                "display_name": ("STRING", {"multiline": False, "default": ""}),
                "description": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, **kwargs):
        return True

    @staticmethod
    def _load_audio_from_bytes(audio_bytes, source_url=""):
        import torch

        ext = ""
        if source_url:
            lower = source_url.lower().split("?")[0]
            for e in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
                if lower.endswith(e):
                    ext = e
                    break
        if not ext:
            header = audio_bytes[:4]
            if header[:3] == b"ID3" or header[:2] == b"\xff\xfb":
                ext = ".mp3"
            elif header[:4] == b"RIFF":
                ext = ".wav"
            elif header[:4] == b"fLaC":
                ext = ".flac"
            elif header[:4] == b"OggS":
                ext = ".ogg"
            else:
                ext = ".wav"

        tmp_path = os.path.join(tempfile.mkdtemp(), f"input_audio{ext}")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        # Method 1: torchaudio from disk file
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(tmp_path)
            print(f"[ExternalAudioFix] Loaded via torchaudio disk: {waveform.shape}, sr={sample_rate}")
            return waveform, sample_rate
        except Exception as e:
            print(f"[ExternalAudioFix] torchaudio disk failed: {e}")

        # Method 2: torchaudio BytesIO with format hint
        try:
            import torchaudio
            audio_io = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_io, format=ext.lstrip("."))
            print(f"[ExternalAudioFix] Loaded via torchaudio BytesIO: {waveform.shape}, sr={sample_rate}")
            return waveform, sample_rate
        except Exception as e:
            print(f"[ExternalAudioFix] torchaudio BytesIO failed: {e}")

        # Method 3: soundfile
        try:
            import soundfile as sf
            import numpy as np
            audio_io = io.BytesIO(audio_bytes)
            data, sample_rate = sf.read(audio_io)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            elif data.ndim == 2:
                data = data.T
            waveform = torch.from_numpy(data).float()
            print(f"[ExternalAudioFix] Loaded via soundfile: {waveform.shape}, sr={sample_rate}")
            return waveform, sample_rate
        except Exception as e:
            print(f"[ExternalAudioFix] soundfile failed: {e}")

        # Method 4: ffmpeg convert to wav then load
        try:
            import subprocess
            out_path = os.path.join(tempfile.mkdtemp(), "converted.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "44100", "-ac", "1", "-f", "wav", out_path],
                capture_output=True, timeout=30
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 44:
                import torchaudio
                waveform, sample_rate = torchaudio.load(out_path)
                print(f"[ExternalAudioFix] Loaded via ffmpeg: {waveform.shape}, sr={sample_rate}")
                return waveform, sample_rate
        except Exception as e:
            print(f"[ExternalAudioFix] ffmpeg failed: {e}")

        raise RuntimeError(f"[ExternalAudioFix] All methods failed. ext={ext}, size={len(audio_bytes)}")

    def load_audio(self, input_id, audio_file, default_value=None, display_name=None, description=None):
        if not audio_file or audio_file.strip() == "":
            if default_value is not None:
                return (default_value,)
            raise ValueError(f"[ExternalAudioFix] No audio for '{input_id}' and no default.")

        try:
            if audio_file.startswith(("http://", "https://")):
                import requests
                print(f"[ExternalAudioFix] Downloading: {audio_file[:80]}...")
                resp = requests.get(audio_file, timeout=60)
                resp.raise_for_status()
                print(f"[ExternalAudioFix] Downloaded {len(resp.content)} bytes")
                waveform, sample_rate = self._load_audio_from_bytes(resp.content, audio_file)
            else:
                import torchaudio
                audio_path = get_annotated_filepath(audio_file)
                waveform, sample_rate = torchaudio.load(audio_path)

            return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)
        except Exception as e:
            print(f"[ExternalAudioFix] ERROR: {e}")
            if default_value is not None:
                return (default_value,)
            raise


NODE_CLASS_MAPPINGS = {"ComfyUIDeployExternalAudio": ComfyUIDeployExternalAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyUIDeployExternalAudio": "External Audio (ComfyUI Deploy)"}
