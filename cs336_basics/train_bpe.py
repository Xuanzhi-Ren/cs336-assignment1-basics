import regex as re

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

def merge_encoded_tokens_with_counter(
    splitted_encoded_tokens: list[list[bytes]],
    special_encoded_tokens: list[bytes],
    counter: dict[tuple[bytes, bytes], int], # the input counter has counted the frequency of each token pair in the list of splitted tokens without special tokens
) -> tuple[list[list[bytes]], dict[tuple[bytes, bytes], int], (tuple[bytes, bytes] | None)]:
    """
    Merges the most frequent pairs of tokens in the splitted encoded tokens and update the counter, while keeping special tokens intact.
    """
    max_freq_pair = None
    max_freq = 0
    # Find the most frequent pair of tokens that are not special tokens
    for pair, freq in counter.items():
        if pair[0] not in special_encoded_tokens and pair[1] not in special_encoded_tokens and (freq > max_freq or (freq == max_freq and (not max_freq_pair or pair > (max_freq_pair[0], max_freq_pair[1])))): # If there are multiple pairs with the same frequency, we choose the one that is lexicographically larger.
            max_freq = freq
            max_freq_pair = pair
    for encoded_tokens in splitted_encoded_tokens:
        i = 0
        # Merge the most frequent pair of tokens in the splitted encoded tokens and update the counter
        while i < len(encoded_tokens) - 1:
            pair = (encoded_tokens[i], encoded_tokens[i + 1])
            if pair == max_freq_pair:
                if i < len(encoded_tokens) - 2:
                    counter[(encoded_tokens[i + 1], encoded_tokens[i + 2])] -= 1
                    if counter[(encoded_tokens[i + 1], encoded_tokens[i + 2])] == 0:
                        del counter[(encoded_tokens[i + 1], encoded_tokens[i + 2])]
                if i > 0:
                    counter[(encoded_tokens[i - 1], encoded_tokens[i])] -= 1
                    if counter[(encoded_tokens[i - 1], encoded_tokens[i])] == 0:
                        del counter[(encoded_tokens[i - 1], encoded_tokens[i])]
                encoded_tokens[i] = pair[0] + pair[1]
                del encoded_tokens[i + 1]
                counter[pair] -= 1
                if counter[pair] == 0:
                    del counter[pair]
                if i > 0:
                    prev_pair = (encoded_tokens[i - 1], encoded_tokens[i])
                    counter[prev_pair] = counter.get(prev_pair, 0) + 1
                if i < len(encoded_tokens) - 1:
                    next_pair = (encoded_tokens[i], encoded_tokens[i + 1])
                    counter[next_pair] = counter.get(next_pair, 0) + 1
            i += 1
    return splitted_encoded_tokens, counter, max_freq_pair
    
def train_bpe(
    input_text: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a BPE tokenizer on the input text, returning the learned vocabulary and the ordered merges.
    """
    presplitted_tokens = split_by_special_tokens(input_text, special_tokens)
    pretokenized_tokens = pretokenize(presplitted_tokens, special_tokens)
    # split the pretokenized tokens into individual characters, while keeping special tokens intact
    splitted_encoded_tokens = []
    for token in pretokenized_tokens:
        if token in special_tokens:
            splitted_encoded_tokens.append([token.encode('utf-8')])
        else:
            encoded_token = token.encode('utf-8')
            splitted_encoded_tokens.append([bytes([b]) for b in encoded_token])
    special_encoded_tokens = [token.encode('utf-8') for token in special_tokens]
    counter: dict[tuple[bytes, bytes], int] = {}
    # Count the frequency of each token pair in the list of splitted tokens without special tokens
    for encoded_tokens in splitted_encoded_tokens:
        for i in range(len(encoded_tokens) - 1):
            pair = (encoded_tokens[i], encoded_tokens[i + 1])
            if pair[0] in special_encoded_tokens or pair[1] in special_encoded_tokens:
                continue
            counter[pair] = counter.get(pair, 0) + 1
    vocab = {}
    merges = []
    # Add all 256 single byte tokens and special tokens to the vocabulary
    for i in range(256):
        token = bytes([i])
        vocab[len(vocab)] = token
    for token in special_encoded_tokens:
        if token not in vocab.values():
            vocab[len(vocab)] = token
    # Merge the most frequent pairs of tokens until the vocabulary size is reached
    while len(vocab) < vocab_size:
        merged_encoded_tokens, updated_counter, max_pair = merge_encoded_tokens_with_counter(
            splitted_encoded_tokens,
            special_encoded_tokens,
            counter
        )
        if not max_pair:
            break
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append((max_pair[0], max_pair[1]))
        splitted_encoded_tokens = merged_encoded_tokens
        counter = updated_counter
    return vocab, merges


def main():
    """
    Test.
    """
    input_text = "a a a a <BOS> Let's see how it works."
    vocab_size = 265
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    vocab, merges = train_bpe(input_text, vocab_size, special_tokens)
    print("Vocabulary:")
    for idx, token in vocab.items():
        print(f"{idx}: {token}")
    print("\nMerges:")
    for merge in merges:
        print(f"{merge[0]} + {merge[1]}")

if __name__ == "__main__":
    main()