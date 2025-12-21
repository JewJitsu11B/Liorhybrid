"""
Cognitive Tokenizer

Handles text tokenization for the cognitive field system.

Uses BPE (Byte Pair Encoding) with special tokens for multimodal inputs.

Special tokens:
- <|text|>: Text mode
- <|image|>: Image follows
- <|video|>: Video follows
- <|endoftext|>: Sequence terminator
- <|pad|>: Padding token
"""

import json
import os
from typing import List, Optional, Union
from pathlib import Path


class CognitiveTokenizer:
    """
    Simple BPE tokenizer with multimodal support.

    For now, uses character-level tokenization as placeholder.
    In production, would use sentencepiece or HuggingFace tokenizers.

    Attributes:
        vocab_size: Total vocabulary size
        special_tokens: Dict of special token IDs
        vocab: Mapping from token string to ID
        inverse_vocab: Mapping from ID to token string
    """

    SPECIAL_TOKENS = {
        '<|pad|>': 0,
        '<|endoftext|>': 1,
        '<|text|>': 2,
        '<|image|>': 3,
        '<|video|>': 4,
        '<|unk|>': 5,
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        vocab_file: Optional[str] = None
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Target vocabulary size
            vocab_file: Optional path to pre-trained vocab file
        """
        self.vocab_size = vocab_size
        self.special_tokens = self.SPECIAL_TOKENS.copy()

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            # Build simple character-level vocab
            self._build_char_vocab()

        # Create inverse mapping
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _build_char_vocab(self):
        """
        Build character-level vocabulary (placeholder).

        In production, would train BPE on corpus.
        """
        # Start with special tokens
        self.vocab = self.special_tokens.copy()

        # Add ASCII characters
        idx = len(self.special_tokens)
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            self.vocab[char] = idx
            idx += 1

        # Add common digrams/trigrams (simplified BPE)
        common_pairs = [
            'th', 'he', 'in', 'er', 'an', 'on', 'at', 'en', 'ed', 'ing',
            'the', 'and', 'ion', 'tion', 'for', 'that', 'with'
        ]

        for pair in common_pairs:
            if idx < self.vocab_size:
                self.vocab[pair] = idx
                idx += 1

        # Fill remaining slots with placeholder tokens
        while idx < self.vocab_size:
            self.vocab[f'<|extra_{idx}|>'] = idx
            idx += 1

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Prepend <|text|> token
            max_length: Optional max sequence length

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.special_tokens['<|text|>'])

        # Greedy tokenization (match longest first)
        i = 0
        while i < len(text):
            matched = False

            # Try matching longer substrings first (3, 2, 1 chars)
            for length in [3, 2, 1]:
                if i + length <= len(text):
                    substr = text[i:i+length]
                    if substr in self.vocab:
                        tokens.append(self.vocab[substr])
                        i += length
                        matched = True
                        break

            if not matched:
                # Unknown character
                tokens.append(self.special_tokens['<|unk|>'])
                i += 1

        if add_special_tokens:
            tokens.append(self.special_tokens['<|endoftext|>'])

        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        return tokens

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Single sequence or batch of sequences
            skip_special_tokens: Remove special tokens from output

        Returns:
            Decoded text string or list of strings
        """
        # Handle batch
        if isinstance(token_ids[0], list):
            return [self.decode(seq, skip_special_tokens) for seq in token_ids]

        # Single sequence
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]

                # Skip special tokens if requested
                if skip_special_tokens and token.startswith('<|') and token.endswith('|>'):
                    continue

                tokens.append(token)

        return ''.join(tokens)

    def save_vocab(self, vocab_file: str):
        """
        Save vocabulary to file.

        Args:
            vocab_file: Path to save vocab JSON
        """
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=2)

    def load_vocab(self, vocab_file: str):
        """
        Load vocabulary from file.

        Args:
            vocab_file: Path to vocab JSON file
        """
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.special_tokens['<|pad|>']

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.special_tokens['<|endoftext|>']

    @property
    def text_token_id(self) -> int:
        """Text mode token ID."""
        return self.special_tokens['<|text|>']

    @property
    def image_token_id(self) -> int:
        """Image mode token ID."""
        return self.special_tokens['<|image|>']

    @property
    def video_token_id(self) -> int:
        """Video mode token ID."""
        return self.special_tokens['<|video|>']


def train_tokenizer_from_corpus(
    corpus_files: List[str],
    vocab_size: int = 32000,
    output_file: str = 'tokenizer_vocab.json'
):
    """
    Train BPE tokenizer on text corpus.

    This is a placeholder. In production, use:
    - sentencepiece
    - HuggingFace tokenizers
    - tiktoken

    Args:
        corpus_files: List of text file paths
        vocab_size: Target vocabulary size
        output_file: Where to save trained vocab
    """
    # For now, just create a basic tokenizer
    tokenizer = CognitiveTokenizer(vocab_size=vocab_size)
    tokenizer.save_vocab(output_file)

    return tokenizer
