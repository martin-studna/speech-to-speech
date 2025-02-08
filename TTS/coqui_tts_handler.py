from threading import Thread
from time import perf_counter
from baseHandler import BaseHandler
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
)
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
import librosa
import logging
from rich.console import Console
from utils.utils import next_power_of_2
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
)

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


torch._inductor.config.fx_graph_cache = True
# mind about this parameter ! should be >= 2 * number of padded prompt sizes for TTS
torch._dynamo.config.cache_size_limit = 15

logger = logging.getLogger(__name__)

console = Console()


if not is_flash_attn_2_available() and torch.cuda.is_available():
    logger.warn(
        """Parler TTS works best with flash attention 2, but is not installed
        Given that CUDA is available in this system, you can install flash attention 2 with `uv pip install flash-attn --no-build-isolation`"""
    )


class CoquiTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        model_name="ylacombe/parler-tts-mini-jenny-30H",
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        gen_kwargs={},
        max_prompt_pad_length=8,
        description=(
            "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. "
            "She speaks very fast."
        ),
        play_steps_s=1,
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.gen_kwargs = gen_kwargs
        self.compile_mode = compile_mode
        self.max_prompt_pad_length = max_prompt_pad_length
        self.description = description

        self.description_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
        config = XttsConfig()
        config.load_json("./assets/config.json")
        self.config = config
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="./assets/")
        self.model.to(self.device)

        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=["assets/samples/en_sample.wav"])
        self.speaker_embedding = speaker_embedding
        self.gpt_cond_latent = gpt_cond_latent
        
        self.blocksize = blocksize

        if self.compile_mode not in (None, "default"):
            logger.warning(
                "Torch compilation modes that captures CUDA graphs are not yet compatible with the TTS part. Reverting to 'default'"
            )
            self.compile_mode = "default"
        
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(
                self.model.forward, mode=self.compile_mode, fullgraph=True
            )

        self.warmup()


    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1 if self.compile_mode == "default" else 2

        if self.device == "cuda":
            torch.cuda.synchronize()
            start_event.record()
        
        for _ in range(n_steps):
            _ = self.model.inference_stream(
                text="Hello, how are you?",
                language="en",
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
            )

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple):
            llm_sentence, _ = llm_sentence
            
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        nb_tokens = len(self.prompt_tokenizer(llm_sentence).input_ids)

        pad_args = {}
        if self.compile_mode:
            # pad to closest upper power of two
            pad_length = next_power_of_2(nb_tokens)
            logger.debug(f"padding to {pad_length}")
            pad_args["pad"] = True
            pad_args["max_length_prompt"] = pad_length


        streamer = self.model.inference_stream(
            text=llm_sentence,
            language="en",
            gpt_cond_latent=self.gpt_cond_latent,
            speaker_embedding=self.speaker_embedding,
        )

        torch.manual_seed(0)
        

        for i, audio_chunk in enumerate(streamer):
            global pipeline_start
            if i == 0 and "pipeline_start" in globals():
                logger.info(
                    f"Time to first audio: {perf_counter() - pipeline_start:.3f}"
                )
            audio_chunk = librosa.resample(audio_chunk.cpu().numpy(), orig_sr=self.config.audio.sample_rate, target_sr=14000)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            for i in range(0, len(audio_chunk), self.blocksize):
                yield np.pad(
                    audio_chunk[i : i + self.blocksize],
                    (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
                )

        self.should_listen.set()
