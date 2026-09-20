import logging
import os
import pathlib

import numpy as np
import sentencepiece

import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48, *, model_path: str | os.PathLike[str] | None = None):
        self._max_len = max_len

        path = (
            pathlib.Path(model_path)
            if model_path is not None
            else download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        )
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # Preserve newlines so that instruction and text2 have a real boundary token
        cleaned_text = prompt.strip().replace("_", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)

    def decode(self, ids: np.ndarray | list[int]) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self._tokenizer.decode(ids)

