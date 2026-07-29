import regex as re
import os
from typing import Self
from collections.abc import Iterable, Iterator
import json
from . import train_bpe

class Tokenizer:
    """A byte-level BPE tokenizer.

    Args:
        vocab: Mapping from token ID to the bytes represented by that token.
        merges: BPE merge rules in increasing rank (priority) order.
        special_tokens: Strings that must be encoded atomically.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Initialize the tokenizer with the reversed dict self.bytes_to_id of vocab, ranked merge pairs self.merge_ranks, etc.
        """
        self.vocab = vocab
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_tokens_to_id = {}

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.bytes_to_id:
                raise ValueError(f"Special token {special_token!r} is not in vocab")
            self.special_tokens_to_id[special_token] = self.bytes_to_id[special_bytes]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None
    ) -> Self:
        """
        Initialize the tokenizer from the filepath of a serialized vocabulary and list of merges
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            serialized_vocab = json.load(f)
        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in serialized_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                left_hex, right_hex = line.split()
                merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(
        self,
        pretoken: str,
    ) -> list[int]:
        """
        Encode a single pretoken that is not a special token into a list of token ids.
        """
        pretoken_bytes = [bytes([byte_value]) for byte_value in pretoken.encode("utf-8")]
        while len(pretoken_bytes) >= 2:
            # Find the pair with least merge rank.
            best_pair = min(
                (pair for pair in zip(pretoken_bytes, pretoken_bytes[1:]) if pair in self.merge_ranks),
                key = self.merge_ranks.__getitem__,
                default = None
                )
            if best_pair is None:
                break
            merged_pretoken_bytes: list[bytes] = []
            i = 0
            while i < len(pretoken_bytes):
                if i + 1 < len(pretoken_bytes) and (pretoken_bytes[i], pretoken_bytes[i + 1]) == best_pair:
                    merged_pretoken_bytes.append(pretoken_bytes[i] + pretoken_bytes[i + 1])
                    i += 2
                else: 
                    merged_pretoken_bytes.append(pretoken_bytes[i])
                    i += 1
            pretoken_bytes = merged_pretoken_bytes

        token_ids = [self.bytes_to_id[token] for token in pretoken_bytes]
        return token_ids

    def encode(
        self,
        text: str
    ) -> list[int]:
        """
        Encode a whole text string into a list of token ids.
        """
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True) # Sort by length in descending order so overlapping special tokens match the longest one first.
        splitted_text = train_bpe.split_by_special_tokens(text, sorted_special_tokens) 
        pretokens = train_bpe.pretokenize(splitted_text, sorted_special_tokens)
        text_ids = []
        for pretoken in pretokens:
            if pretoken in sorted_special_tokens:
                text_ids.append(self.bytes_to_id[pretoken.encode("utf-8")])
            else:
                text_ids.extend(self._encode_pretoken(pretoken))
        return text_ids 

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int] :
        """
        Encode an iterable of strings by lazily yielding token ids.
        """
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(
        self,
        ids: list[int]
    ) -> str:
        """
        Decode a whole token id list into a text string.
        """
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")
    
        
