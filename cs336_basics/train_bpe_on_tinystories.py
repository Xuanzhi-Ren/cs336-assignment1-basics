import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import BinaryIO

import regex as re


# Worker processes initially return:
#
#     UTF-8 bytes of a pre-token -> occurrence count
#
# Using bytes as keys reduces inter-process communication and memory.
BytePretokenCounter = Counter[bytes]

# The BPE merge stage operates on sequences of token IDs.
TokenSequence = tuple[int, ...]
SequenceCounter = Counter[TokenSequence]

def split_by_special_tokens(
    input_text: str,
    special_tokens: list[str],
) -> list[str]:
    """
    Splits the input text into a list of tokens, separating out any special tokens.
    """
    if not special_tokens: # Corner case: if special_tokens is empty, re.split(pattern, input_text) will return a list of single characters.
        return [input_text]
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "(" + "|".join(escaped_tokens) + ")"
    tokens = re.split(pattern, input_text)
    return [token for token in tokens if token]  # Remove empty strings

def pretokenize(
    input_tokens: list[str],
    special_tokens: list[str]
) -> list[str]:
    """
    Pre-tokenizes the input tokens by the pattern given in the pdf.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenized_tokens = []
    for token in input_tokens:
        if token in special_tokens:
            pretokenized_tokens.append(token)
        else:
            pretokenized_tokens.extend(re.findall(PAT, token))
    return pretokenized_tokens

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    This function is from pretokenization_example.py.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def initialize_worker(
    input_path: str,
    special_tokens: tuple[str, ...],
) -> None:
    """
    初始化 multiprocessing 子进程。

    通过 initializer 设置一次全局变量，可以避免在每个任务中
    重复传输文件路径、特殊 token 和正则表达式。
    """
    global _WORKER_INPUT_PATH
    global _WORKER_SPECIAL_TOKENS
    global _WORKER_SPECIAL_SPLIT_PATTERN
    global _WORKER_DECODE_ERRORS

    _WORKER_INPUT_PATH = input_path

    _WORKER_SPECIAL_TOKENS = special_tokens



def count_chunk_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    byte_range: tuple[int, int],
) -> BytePretokenCounter:
    """
    Counts the frequency of pre-tokens (exclude special tokens) in a chunk of the file specified by byte_range.
    """
    start, end = byte_range
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    text = chunk.decode("utf-8")
    local_counter = Counter()
    splitted_text = split_by_special_tokens(text, special_tokens)
    for token in pretokenize(splitted_text, special_tokens):
        if token in special_tokens:
            continue
        local_counter[token.encode("utf-8")] += 1
    return local_counter

def collect_pretoken_counts_parallel(
    input_path: str | os.PathLike[str],
    special_tokens: list[str],
    split_special_token: bytes,
    num_processes: int,
    num_chunks: int,
) -> BytePretokenCounter:
    """
    Collects the frequency of pre-tokens in the input file in parallel.
    """
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_chunks, split_special_token)
    byte_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    actual_process = min(num_processes, len(byte_ranges))
    global_counter = Counter()
    with mp.Pool(processes=actual_process) as pool:
        worker = partial(count_chunk_pretokens, input_path, special_tokens)
        results = pool.imap_unordered(worker, byte_ranges, chunksize=1)
        for _, local_counter in enumerate(results, start=1):
            global_counter.update(local_counter)
    return global_counter

def convert_byte_counter_to_sequence(
    byte_counter: BytePretokenCounter,
) -> SequenceCounter:
    """
    Convert counter with byte pretokens into counter with int sequences pretokens.
    For example, b"abc" -> (97, 98, 99)
    """
    seq_counter = Counter()
    while byte_counter:
        encoded_pretoken, frequency = byte_counter.popitem()
        seq_counter[tuple(encoded_pretoken)] = frequency
    return seq_counter

def count_all_pairs(
    seq_counter: SequenceCounter
) -> Counter[tuple[int, int]]:
    """
    Count all adjacent pairs in each pretokens. 
    Keep the frequency of each pretokens in seq_counter as the weight of pairs.
    """
    pair_counter : Counter[tuple[int, int]] = Counter()
    for seq, frequency in seq_counter.items():
        pair_count = 0
        for i in range(len(seq) - 1):
            pair_counter[(seq[i], seq[i + 1])] += frequency
    return pair_counter

def merge_pair_in_sequence(
    seq : TokenSequence,
    pair_to_merge : tuple[int, int],
    new_token_id : int,
) -> TokenSequence:
    """
    Merge the most frequent pair in the sequence from left to right:
        e.g. (a, a, a) -> (aa, a)
             (97, 97, 97) -> (256, 97)
    """
    merged_seq: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and pair_to_merge == (seq[i], seq[i + 1]):
            merged_seq.append(new_token_id)
            i += 2
        else:
            merged_seq.append(seq[i])
            i += 1
    return tuple(merged_seq)

def train_bpe_from_pretoken_counter(
    pretoken_counter: BytePretokenCounter,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a BPE tokenizer on the pretoken counter, returning the learned vocabulary and the ordered merges.
    """
    # Initialize the 256 single-byte tokens and spectial_tokens.
    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    if len(vocab) > vocab_size:
        print("Warning: initial vocabulary exceeded the vocab_size limit")
    seq_counter = convert_byte_counter_to_sequence(pretoken_counter)
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pair_counter = count_all_pairs(seq_counter)
        # First, select the pair with the highest frequency.
        # If multiple pairs have the same frequency, select the pair with the lexicographically greatest corresponding byte sequences.
        left_token_id, right_token_id = max(pair_counter, key=lambda pair: (pair_counter[pair], vocab[pair[0]], vocab[pair[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[left_token_id] + vocab[right_token_id]
        merges.append((vocab[left_token_id], vocab[right_token_id]))
        # update sequence counter
        updated_seq_counter:SequenceCounter = Counter()
        while seq_counter:
            seq, frequency = seq_counter.popitem()
            merged_seq = merge_pair_in_sequence(seq, pair_to_merge=(left_token_id, right_token_id), new_token_id= new_id)
            updated_seq_counter[merged_seq] += frequency
        seq_counter = updated_seq_counter
    return vocab, merges

def train_bpe_file(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: bytes = b"<|endoftext|>",
    num_processes: int | None = None,
    num_chunks: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
        Train BPE on a large text file.

        Parallelized steps:
            Reading file chunks, splitting on special tokens, pre-tokenization, and counting pre-tokens.

        Sequential steps:
            Merging the Counters from all chunks and performing global BPE merges one at a time.
    """
    if num_processes is None:
        num_processes = min(8, os.cpu_count() or 1)
    if num_chunks is None:
        num_chunks = num_processes * 8
    byte_pretoken_counter: BytePretokenCounter = collect_pretoken_counts_parallel(input_path, special_tokens, split_special_token, num_processes, num_chunks)
    return train_bpe_from_pretoken_counter(byte_pretoken_counter, vocab_size, special_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories.")
    parser.add_argument("--input", dest="input_path", type=Path, required=True, help="Path to the training corpus.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for vocab.json and merge.txt.")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Final vocabulary size, including special tokens.")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of pre-tokenization worker processes.")
    parser.add_argument("--num-chunks", type=int, default=64, help="Number of corpus chunks to process.")
    args = parser.parse_args()

    if args.vocab_size < 257:
        parser.error("--vocab-size must be at least 257 (256 byte tokens plus <|endoftext|>).")
    if args.num_processes < 1:
        parser.error("--num-processes must be at least 1.")
    if args.num_chunks < 1:
        parser.error("--num-chunks must be at least 1.")
    if not args.input_path.is_file():
        parser.error(f"input file does not exist: {args.input_path}")

    return args


def main() -> None:
    args = parse_args()
    special_tokens = ["<|endoftext|>"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    vocab, merges = train_bpe_file(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        split_special_token=b"<|endoftext|>",
        num_processes=args.num_processes,
        num_chunks=args.num_chunks,
    )
    elapsed_seconds = time.perf_counter() - start_time

    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merge.txt"
    with vocab_path.open("w", encoding="utf-8") as file:
        json.dump(
            {str(token_id): token.hex() for token_id, token in vocab.items()},
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")
    with merges_path.open("w", encoding="utf-8") as file:
        for left_token, right_token in merges:
            file.write(f"{left_token.hex()} {right_token.hex()}\n")

    special_token_bytes = {token.encode("utf-8") for token in special_tokens}
    longest_token_id, longest_token = max(
        ((token_id, token) for token_id, token in vocab.items() if token not in special_token_bytes),
        key=lambda item: len(item[1]),
    )

    print(f"vocab size: {len(vocab)}")
    print(f"merge count: {len(merges)}")
    print(f"training time: {elapsed_seconds:.2f} seconds")
    print(f"longest learned token id: {longest_token_id}")
    print(f"longest learned token bytes: {longest_token!r}")
    print(f"longest learned token length: {len(longest_token)} bytes")
    print(f"saved vocabulary to: {vocab_path}")
    print(f"saved merges to: {merges_path}")


if __name__ == "__main__":
    main()
