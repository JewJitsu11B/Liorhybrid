"""
Cognitive Tokenizer - HuggingFace Backend

Fast, parallel tokenization using HuggingFace tokenizers (Rust-based).

Key features:
- GIL-FREE: Rust backend releases Python GIL during tokenization
- PARALLEL: Batch encoding uses all CPU cores automatically
- BPE: Real byte-pair encoding, not character-level placeholder
- NO LENGTH LIMIT: Works with RoPE embeddings for any sequence length

Special tokens:
- <|text|>: Text mode
- <|image|>: Image follows
- <|video|>: Video follows
- <|endoftext|>: Sequence terminator
- <|pad|>: Padding token
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import json
import os
import re
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path

# Try HuggingFace tokenizers (Rust, GIL-free)
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSequence
    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    HF_TOKENIZERS_AVAILABLE = False
    print("[WARNING] HuggingFace tokenizers not installed. Using fallback.")
    print("         Install with: pip install tokenizers")


class SpanBoundaryDetector:
    """
    Lightweight sentence/phrase boundary detector. No vocab, just indices.

    Uses HF offset mappings (NOT char/token ratio) for correct span detection
    with byte-level tokenizers like GPT-2.
    """

    SENTENCE_END = re.compile(r'[.!?]+(?:\s|$)')
    PHRASE_END = re.compile(r'[,;:]+(?:\s|$)')

    def char_to_token_idx(self, char_pos: int, offsets: List[Tuple[int, int]]) -> int:
        """Convert character position to token index using HF offset mappings."""
        for tok_idx, (start, end) in enumerate(offsets):
            if start <= char_pos < end:
                return tok_idx
            if char_pos < start:
                return tok_idx  # Return the token that starts after this position
        return len(offsets)  # End of sequence

    def find_sentence_spans(
        self,
        text: str,
        offsets: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Return (start, end) token indices for each sentence.

        Uses HF offset mappings for accurate char->token conversion.
        """
        n_tokens = len(offsets)
        if n_tokens == 0:
            return [(0, 0)]

        # Find sentence boundaries in CHARACTER space
        char_boundaries = [0]
        for m in self.SENTENCE_END.finditer(text):
            char_boundaries.append(m.end())
        if char_boundaries[-1] < len(text):
            char_boundaries.append(len(text))

        # Convert char boundaries to TOKEN boundaries using offsets
        token_boundaries = []
        for char_pos in char_boundaries:
            tok_idx = self.char_to_token_idx(char_pos, offsets)
            if not token_boundaries or tok_idx > token_boundaries[-1]:
                token_boundaries.append(tok_idx)

        if not token_boundaries:
            return [(0, n_tokens)]
        if token_boundaries[-1] < n_tokens:
            token_boundaries.append(n_tokens)

        return [
            (token_boundaries[i], token_boundaries[i + 1])
            for i in range(len(token_boundaries) - 1)
            if token_boundaries[i + 1] > token_boundaries[i]
        ]

    def find_phrase_spans(
        self,
        text: str,
        offsets: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Return (start, end) token indices for each phrase (comma-separated).

        Uses HF offset mappings for accurate char->token conversion.
        """
        n_tokens = len(offsets)
        if n_tokens == 0:
            return [(0, 0)]

        # Find phrase boundaries in CHARACTER space (commas, semicolons, colons)
        char_boundaries = [0]
        for m in self.PHRASE_END.finditer(text):
            char_boundaries.append(m.end())
        # Also include sentence boundaries as phrase boundaries
        for m in self.SENTENCE_END.finditer(text):
            char_boundaries.append(m.end())
        char_boundaries = sorted(set(char_boundaries))
        if char_boundaries[-1] < len(text):
            char_boundaries.append(len(text))

        # Convert to token boundaries
        token_boundaries = []
        for char_pos in char_boundaries:
            tok_idx = self.char_to_token_idx(char_pos, offsets)
            if not token_boundaries or tok_idx > token_boundaries[-1]:
                token_boundaries.append(tok_idx)

        if not token_boundaries:
            return [(0, n_tokens)]
        if token_boundaries[-1] < n_tokens:
            token_boundaries.append(n_tokens)

        return [
            (token_boundaries[i], token_boundaries[i + 1])
            for i in range(len(token_boundaries) - 1)
            if token_boundaries[i + 1] > token_boundaries[i]
        ]


class CognitiveTokenizer:
    """
    Fast BPE tokenizer with multimodal support.

    Uses HuggingFace tokenizers library (Rust backend, GIL-free).
    Falls back to simple character-level if tokenizers not installed.

    Key methods:
    - encode(text) -> List[int]: Single text to token IDs
    - encode_batch(texts) -> List[List[int]]: PARALLEL batch encoding
    - decode(ids) -> str: Token IDs to text
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
        vocab_size: int = 50257,  # GPT-2 default vocab size
        vocab_file: Optional[str] = None,
        pretrained: Optional[str] = "gpt2"  # DEFAULT TO GPT-2 (Rust-fast, 50k vocab)
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Target vocabulary size (default: GPT-2's 50257)
            vocab_file: Path to saved tokenizer JSON
            pretrained: Name of pretrained tokenizer to load (default: "gpt2")
        """
        self.vocab_size = vocab_size
        self.special_tokens = self.SPECIAL_TOKENS.copy()
        self._hf_tokenizer = None
        self.span_detector = SpanBoundaryDetector()  # For sentence/phrase span detection

        if HF_TOKENIZERS_AVAILABLE:
            if pretrained:
                self._load_pretrained(pretrained)
            elif vocab_file and os.path.exists(vocab_file):
                self._load_hf_tokenizer(vocab_file)
            else:
                # Fallback to GPT-2 instead of empty BPE
                self._load_pretrained("gpt2")
        else:
            # Fallback to simple vocab
            if vocab_file and os.path.exists(vocab_file):
                self.load_vocab(vocab_file)
            else:
                self._build_char_vocab()
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _load_pretrained(self, name: str):
        """Load pretrained tokenizer from HuggingFace."""
        try:
            from tokenizers import Tokenizer
            self._hf_tokenizer = Tokenizer.from_pretrained(name)
            # Add our special tokens
            self._hf_tokenizer.add_special_tokens(list(self.special_tokens.keys()))
            self.vocab_size = self._hf_tokenizer.get_vocab_size()
        except Exception as e:
            print(f"[WARNING] Could not load pretrained tokenizer '{name}': {e}")
            self._build_hf_tokenizer()

    def _build_hf_tokenizer(self):
        """Build BPE tokenizer with HuggingFace."""
        # Create BPE tokenizer - byte-level BPE doesn't need unk_token
        # since any byte sequence can be represented
        self._hf_tokenizer = Tokenizer(models.BPE(unk_token=None))

        # Normalizer: NFD unicode normalization
        self._hf_tokenizer.normalizer = NormSequence([NFD()])

        # Pre-tokenizer: split on whitespace and punctuation
        self._hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Add special tokens
        special_tokens = list(self.special_tokens.keys())
        self._hf_tokenizer.add_special_tokens(special_tokens)

        # Post-processor for adding special tokens
        self._hf_tokenizer.post_processor = processors.TemplateProcessing(
            single="<|text|> $A <|endoftext|>",
            special_tokens=[
                ("<|text|>", self.special_tokens["<|text|>"]),
                ("<|endoftext|>", self.special_tokens["<|endoftext|>"]),
            ],
        )

        # Enable padding
        self._hf_tokenizer.enable_padding(
            pad_id=self.special_tokens["<|pad|>"],
            pad_token="<|pad|>"
        )

    def _load_hf_tokenizer(self, path: str):
        """Load tokenizer from file."""
        self._hf_tokenizer = Tokenizer.from_file(path)

    def _build_char_vocab(self):
        """Fallback: Build character-level vocabulary."""
        self.vocab = self.special_tokens.copy()
        idx = len(self.special_tokens)

        # ASCII characters
        for i in range(32, 127):
            self.vocab[chr(i)] = idx
            idx += 1

        # Common subwords
        for pair in ['th', 'he', 'in', 'er', 'an', 'on', 'at', 'en', 'ed', 'ing',
                     'the', 'and', 'ion', 'tion', 'for', 'that', 'with']:
            if idx < self.vocab_size:
                self.vocab[pair] = idx
                idx += 1

        # Fill remaining
        while idx < self.vocab_size:
            self.vocab[f'<|extra_{idx}|>'] = idx
            idx += 1

    def train(self, files: List[str], vocab_size: Optional[int] = None):
        """
        Train tokenizer on corpus files.

        Args:
            files: List of text file paths
            vocab_size: Target vocabulary size (default: self.vocab_size)
        """
        if not HF_TOKENIZERS_AVAILABLE:
            print("[WARNING] Cannot train without HuggingFace tokenizers")
            return

        vocab_size = vocab_size or self.vocab_size

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            min_frequency=2,
            show_progress=True
        )

        self._hf_tokenizer.train(files, trainer)
        self.vocab_size = self._hf_tokenizer.get_vocab_size()
        print(f"[Tokenizer] Trained on {len(files)} files, vocab_size={self.vocab_size}")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        return_spans: bool = False,
        span_type: str = "sentence"  # "sentence" or "phrase"
    ) -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
        """
        Encode text to token IDs, optionally with span boundaries.

        Args:
            text: Input text string
            add_special_tokens: Add <|text|> and <|endoftext|> tokens
            max_length: Truncate to this length (None = no limit)
            return_spans: If True, also return sentence/phrase span boundaries
            span_type: Type of spans to return ("sentence" or "phrase")

        Returns:
            If return_spans=False: List of token IDs
            If return_spans=True: Tuple of (token IDs, spans)
                where spans is List[(start_token_idx, end_token_idx)]
        """
        if self._hf_tokenizer is not None:
            # HuggingFace path (fast, GIL-free)
            # Request offset mappings for accurate span detection
            encoding = self._hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
            ids = encoding.ids
            offsets = encoding.offsets  # [(char_start, char_end), ...] for each token

            if max_length is not None and len(ids) > max_length:
                ids = ids[:max_length]
                offsets = offsets[:max_length] if offsets else []

            if not return_spans:
                return ids

            # Use offset mappings for accurate span detection
            if offsets:
                if span_type == "phrase":
                    spans = self.span_detector.find_phrase_spans(text, offsets)
                else:
                    spans = self.span_detector.find_sentence_spans(text, offsets)
            else:
                # Fallback if no offsets available
                spans = [(0, len(ids))]

            return ids, spans
        else:
            # Fallback path
            ids = self._encode_fallback(text, add_special_tokens, max_length)
            if not return_spans:
                return ids
            # Simple fallback: one span for entire sequence
            return ids, [(0, len(ids))]

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[List[int]]:
        """
        PARALLEL batch encoding - GIL-FREE.

        This is the fast path. Uses all CPU cores automatically.

        Args:
            texts: List of text strings
            add_special_tokens: Add special tokens to each
            max_length: Truncate each to this length

        Returns:
            List of token ID lists
        """
        if self._hf_tokenizer is not None:
            # HuggingFace parallel encoding (Rust, releases GIL)
            encodings = self._hf_tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)

            if max_length is not None:
                return [enc.ids[:max_length] for enc in encodings]
            return [enc.ids for enc in encodings]
        else:
            # Fallback (sequential, slow)
            return [self._encode_fallback(t, add_special_tokens, max_length) for t in texts]

    def _encode_fallback(
        self,
        text: str,
        add_special_tokens: bool,
        max_length: Optional[int]
    ) -> List[int]:
        """Fallback encoding (character-level, sequential)."""
        tokens = []

        if add_special_tokens:
            tokens.append(self.special_tokens['<|text|>'])

        i = 0
        while i < len(text):
            matched = False
            for length in [3, 2, 1]:
                if i + length <= len(text):
                    substr = text[i:i+length]
                    if substr in self.vocab:
                        tokens.append(self.vocab[substr])
                        i += length
                        matched = True
                        break
            if not matched:
                tokens.append(self.special_tokens['<|unk|>'])
                i += 1

        if add_special_tokens:
            tokens.append(self.special_tokens['<|endoftext|>'])

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
            token_ids: Single sequence or batch
            skip_special_tokens: Remove special tokens

        Returns:
            Decoded text
        """
        # Handle batch
        if token_ids and isinstance(token_ids[0], list):
            return [self.decode(seq, skip_special_tokens) for seq in token_ids]

        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            # Fallback
            result = []
            for tid in token_ids:
                if tid in self.inverse_vocab:
                    token = self.inverse_vocab[tid]
                    if skip_special_tokens and token.startswith('<|'):
                        continue
                    result.append(token)
            return ''.join(result)

    def save(self, path: str):
        """Save tokenizer to file."""
        if self._hf_tokenizer is not None:
            self._hf_tokenizer.save(path)
        else:
            with open(path, 'w') as f:
                json.dump({'vocab': self.vocab, 'special_tokens': self.special_tokens}, f)

    def save_vocab(self, vocab_file: str):
        """Save vocabulary (compatibility method)."""
        self.save(vocab_file)

    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        with open(vocab_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'vocab' in data:
                self.vocab = data['vocab']
                self.special_tokens = data.get('special_tokens', self.SPECIAL_TOKENS)
            else:
                self.vocab = data

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.get_vocab_size()
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dict."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.get_vocab()
        return self.vocab

    @property
    def vocab(self) -> Dict[str, int]:
        """Vocabulary property for compatibility."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.get_vocab()
        return self._vocab

    @vocab.setter
    def vocab(self, value: Dict[str, int]):
        self._vocab = value

    @property
    def inverse_vocab(self) -> Dict[int, str]:
        """Inverse vocabulary."""
        v = self.vocab
        return {v: k for k, v in v.items()}

    @inverse_vocab.setter
    def inverse_vocab(self, value: Dict[int, str]):
        self._inverse_vocab = value

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens['<|pad|>']

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens['<|endoftext|>']

    @property
    def text_token_id(self) -> int:
        return self.special_tokens['<|text|>']

    @property
    def image_token_id(self) -> int:
        return self.special_tokens['<|image|>']

    @property
    def video_token_id(self) -> int:
        return self.special_tokens['<|video|>']


def train_tokenizer_from_corpus(
    corpus_files: List[str],
    vocab_size: int = 32000,
    output_file: str = 'tokenizer.json'
) -> CognitiveTokenizer:
    """
    Train BPE tokenizer on text corpus.

    Args:
        corpus_files: List of text file paths
        vocab_size: Target vocabulary size
        output_file: Path to save trained tokenizer

    Returns:
        Trained CognitiveTokenizer
    """
    tokenizer = CognitiveTokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus_files, vocab_size=vocab_size)
    tokenizer.save(output_file)
    print(f"[Tokenizer] Saved to {output_file}")
    return tokenizer
