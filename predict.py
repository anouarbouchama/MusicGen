from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch
from cog import BaseModel, BasePredictor, Input, Path

class Output(BaseModel):
    """
    Output class for riffusion predictions
    """

    audio: Path
    spectrogram: Path
    error: T.Optional[str] = None

def predict(duration,prompt):
  model = musicgen.MusicGen.get_pretrained('medium')
  model.set_generation_params(duration=duration)
  prompts=[prompt]
  res = model.generate(prompts, progress=True)
  return display_audio(res, 32000)


duration=5
prompt='jazz'
predict(duration,prompt)
