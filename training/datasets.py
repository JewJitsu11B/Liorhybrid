"""
Multimodal Datasets

Dataset classes for loading and preprocessing multimodal data.

Supported formats:
- Text: Plain text files, JSONL
- Image-Text: COCO-style, WebDataset
- Video-Text: Frame directories, video files
- Q/A: Question-answer pairs
- RLHF: Preference data (future)

Performance features:
- StreamingTextDataset: Lazy loading, doesn't load all data into memory
- Multiprocessing support via num_workers in DataLoader
- Memory-mapped file access for large datasets
"""

import torch
from torch.utils.data import Dataset, IterableDataset
import json
import os
import mmap
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator, Sequence, Union
import random

# Optional dependencies for image/video
#try:
 #   from PIL import Image
  #  VISION_AVAILABLE = True
#except ImportError:
 #   Image = None
  #  VISION_AVAILABLE = False

# Import transforms - REQUIRED for image/video datasets
#if VISION_AVAILABLE:
  #  import torchvision.transforms as transforms


class TextDataset(Dataset):
    """
    Text-only dataset for language modeling.

    Supports:
    - Plain text files (one document per line)
    - JSONL format
    - Sliding window over long documents

    Args:
        data_path: Path to text file or directory
        tokenizer: CognitiveTokenizer instance
        max_length: Maximum sequence length
        stride: Stride for sliding window (if None, use max_length)
    """

    def __init__(
        self,
        data_path: Union[str, Sequence[str]],
        tokenizer,
        max_length: int = 512,
        stride: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length

        # Load data
        self.examples = []
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Load text data from file or directory."""
        path = Path(data_path)

        if path.is_file():
            self._load_file(path)
        elif path.is_dir():
            for file_path in path.glob('**/*.txt'):
                self._load_file(file_path)
            for file_path in path.glob('**/*.jsonl'):
                self._load_jsonl(file_path)
        else:
            raise ValueError(f"Invalid data path: {data_path}")

    def _load_file(self, file_path: Path):
        """Load plain text file (ASCII only)."""
        with open(file_path, 'rb') as f:
            content = f.read()

        # Decode and strip non-ASCII
        text = content.decode('ascii', errors='ignore')

        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Tokenize
                token_ids = self.tokenizer.encode(line, max_length=self.max_length)

                # Create sliding window if text is too long
                if len(token_ids) > self.max_length:
                    for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                        self.examples.append(token_ids[i:i + self.max_length])
                else:
                    self.examples.append(token_ids)

    def _load_jsonl(self, file_path: Path):
        """Load JSONL file (one JSON per line, ASCII only)."""
        with open(file_path, 'rb') as f:
            content = f.read()

        # Decode and strip non-ASCII
        text = content.decode('ascii', errors='ignore')

        for line in text.split('\n'):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    text = data.get('text', data.get('content', ''))
                    if text:
                        token_ids = self.tokenizer.encode(text, max_length=self.max_length)
                        self.examples.append(token_ids)
                except json.JSONDecodeError:
                    continue  # Skip malformed JSON lines

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training example.

        Returns:
            dict with:
            - input_ids: (seq_len,) token IDs
            - labels: (seq_len,) labels (same as input_ids for LM)
            - attention_mask: (seq_len,) mask (all ones)
        """
        token_ids = self.examples[idx]

        # Pad to max_length
        seq_len = len(token_ids)
        if seq_len < self.max_length:
            padding = [self.tokenizer.pad_token_id] * (self.max_length - seq_len)
            token_ids = token_ids + padding
            attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)
        else:
            attention_mask = [1] * self.max_length

        labels = token_ids.copy()
        # Mask out padding positions so cross-entropy ignores them
        for i, m in enumerate(attention_mask):
            if m == 0:
                labels[i] = -100

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),  # For LM loss
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'modality': 'text',
            'modality_id': torch.tensor(0, dtype=torch.long)  # text=0, image=1, video=2, audio=3
        }


# Modality ID constants for future multimodal extension
MODALITY_TEXT = 0
MODALITY_IMAGE = 1
MODALITY_VIDEO = 2
MODALITY_AUDIO = 3


class StreamingTextDataset(IterableDataset):
    """
    Streaming text dataset - LAZY LOADING for large files.

    Does NOT load all data into memory. Reads and tokenizes on-the-fly.
    Use with DataLoader(num_workers=20) for datasets > 100MB.

    Args:
        data_path: Path to text file or directory
        tokenizer: CognitiveTokenizer instance
        max_length: Maximum sequence length
        shuffle_buffer: Size of shuffle buffer (0 = no shuffle)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        shuffle_buffer: int = 10000,
        seed: int = 42
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        # Collect file paths only (fast)
        self.file_paths = []
        if self.data_path.is_file():
            self.file_paths = [self.data_path]
        elif self.data_path.is_dir():
            self.file_paths = list(self.data_path.glob('**/*.txt'))
            self.file_paths += list(self.data_path.glob('**/*.jsonl'))

        # Check total size for worker recommendation
        total_bytes = sum(f.stat().st_size for f in self.file_paths)
        total_mb = total_bytes / (1024 * 1024)
        recommended_workers = 20 if total_mb > 100 else 12
        print(f"[StreamingTextDataset] {len(self.file_paths)} files, {total_mb:.1f}MB total")
        print(f"[StreamingTextDataset] Recommended: DataLoader(num_workers={recommended_workers})")

    def _make_example(self, token_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Convert token IDs to training example with padding."""
        seq_len = len(token_ids)
        if seq_len < self.max_length:
            padding = [self.tokenizer.pad_token_id] * (self.max_length - seq_len)
            token_ids = token_ids + padding
            attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)
        else:
            token_ids = token_ids[:self.max_length]
            attention_mask = [1] * self.max_length

        labels = token_ids.copy()
        # Mask out padding positions so cross-entropy ignores them
        for i, m in enumerate(attention_mask):
            if m == 0:
                labels[i] = -100

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'modality': 'text',
            'modality_id': torch.tensor(MODALITY_TEXT, dtype=torch.long)
        }

    def _stream_file(self, file_path: Path):
        """Stream examples from a single file."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            text = content.decode('utf-8', errors='ignore')

            if file_path.suffix == '.jsonl':
                for line in text.split('\n'):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        text_content = data.get('text', data.get('content', ''))
                        if text_content and len(text_content) >= 10:
                            token_ids = self.tokenizer.encode(text_content, max_length=self.max_length * 2)  # Allow longer for sliding window
                            
                            # Handle long sequences with sliding window
                            if len(token_ids) > self.max_length:
                                stride = self.max_length // 2
                                for i in range(0, len(token_ids) - self.max_length + 1, stride):
                                    window_tokens = token_ids[i:i + self.max_length]
                                    if len(window_tokens) >= 5:
                                        yield self._make_example(window_tokens)
                            elif len(token_ids) >= 5:
                                yield self._make_example(token_ids)
                    except json.JSONDecodeError:
                        continue
            else:
                for line in text.split('\n'):
                    line = line.strip()
                    if line and len(line) >= 10:
                        token_ids = self.tokenizer.encode(line, max_length=self.max_length * 2)  # Allow longer for sliding window
                        
                        # Handle long sequences with sliding window
                        if len(token_ids) > self.max_length:
                            stride = self.max_length // 2
                            for i in range(0, len(token_ids) - self.max_length + 1, stride):
                                window_tokens = token_ids[i:i + self.max_length]
                                if len(window_tokens) >= 5:
                                    yield self._make_example(window_tokens)
                        elif len(token_ids) >= 5:
                            yield self._make_example(token_ids)
        except Exception as e:
            print(f"[StreamingTextDataset] Error reading {file_path}: {e}")

    def __iter__(self):
        """Iterate through files, handling multiprocessing."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.file_paths
            worker_seed = self.seed
        else:
            # Split files across workers
            per_worker = len(self.file_paths) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.file_paths)
            files = self.file_paths[start:end]
            worker_seed = self.seed + worker_info.id

        rng = random.Random(worker_seed)
        files = list(files)
        rng.shuffle(files)

        if self.shuffle_buffer > 0:
            buffer = []
            for file_path in files:
                for example in self._stream_file(file_path):
                    buffer.append(example)
                    if len(buffer) >= self.shuffle_buffer:
                        rng.shuffle(buffer)
                        yield from buffer
                        buffer = []
            if buffer:
                rng.shuffle(buffer)
                yield from buffer
        else:
            for file_path in files:
                yield from self._stream_file(file_path)


# ==============================================================================
# STREAMING INFRASTRUCTURE (three orthogonal concerns)
# ==============================================================================

class StreamContext:
    """Metadata for stream hooks - cheap, explicit, future-proof."""
    __slots__ = ("file", "worker_id", "byte_start", "byte_end")

    def __init__(self, *, file: Path, worker_id: int, byte_start: int, byte_end: int):
        self.file = file
        self.worker_id = worker_id
        self.byte_start = byte_start
        self.byte_end = byte_end


class StreamHooks:
    """
    Observational hooks for streaming datasets.
    Rules: hooks never return values, never mutate inputs, zero overhead when unused.
    """
    def on_line(self, *, text: str, ctx: StreamContext) -> None:
        pass

    def on_tokens(self, *, tokens: List[int], ctx: StreamContext) -> None:
        pass

    def on_example(self, *, example: Dict[str, torch.Tensor], ctx: StreamContext) -> None:
        pass


class NullStreamHooks(StreamHooks):
    """No-op hooks for maximum performance."""
    __slots__ = ()
    pass


class TokenBlockBuilder:
    """Stateless text -> token block builder. Centralizes all tokenization logic."""

    def __init__(self, tokenizer, *, max_length: int, min_seq_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_seq_length = min_seq_length

    def extract_text(self, raw: str, *, suffix: str) -> Optional[str]:
        """Extract text from raw line based on file type."""
        if suffix == ".jsonl":
            try:
                obj = json.loads(raw)
                return obj.get("text") or obj.get("content")
            except Exception:
                return None
        return raw

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text without truncation."""
        return self.tokenizer.encode(text, max_length=None)

    def yield_blocks(self, tokens: List[int]) -> Iterator[List[int]]:
        """Yield fixed-length token blocks from a token list."""
        buf = list(tokens)
        while len(buf) >= self.max_length:
            block = buf[:self.max_length]
            buf = buf[self.max_length:]
            if len(block) >= self.min_seq_length:
                yield block
        if len(buf) >= self.min_seq_length:
            yield buf


def iter_lines_in_byte_range(
    file_path: Path,
    *,
    start: int,
    end: int,
    read_chunk_bytes: int
) -> Iterator[str]:
    """
    Iterate lines in a byte range of a file.
    Pure byte->line producer. No tokenization, no hooks, no policy.
    """
    if end <= start:
        return

    with open(file_path, "rb") as f:
        f.seek(start)
        buf = b""
        read_bytes = 0
        max_bytes = end - start

        while read_bytes < max_bytes:
            to_read = min(read_chunk_bytes, max_bytes - read_bytes)
            chunk = f.read(to_read)
            if not chunk:
                break

            read_bytes += len(chunk)
            buf += chunk

            parts = buf.split(b"\n")
            buf = parts.pop() if parts else b""

            for raw in parts:
                if raw:
                    yield raw.decode("utf-8", errors="ignore").strip()

        if buf:
            yield buf.decode("utf-8", errors="ignore").strip()


# ==============================================================================
# CHUNKED TEXT DATASET (best of both worlds)
# ==============================================================================

class ChunkedTextDataset(IterableDataset):
    """
    Global byte-range streaming dataset with hooks.

    Properties:
    - Byte-range parallelism preserved
    - Streaming safety preserved
    - Token logic centralized via TokenBlockBuilder
    - Hooks are explicit, cheap, and future-safe
    - Memory, SDM, logging, analytics attach cleanly
    - No recursion, no snapshotting, no accidental coupling

    Its only job: "Turn bytes into sequences as fast as physics allows."
    """

    def __init__(
        self,
        data_path: Union[str, Sequence[str]],
        tokenizer,
        *,
        max_length: int = 512,
        chunk_size: int = 10000,  # API compat, ignored
        bptt_window: int = 50,    # API compat, ignored
        min_seq_length: int = 4,  # Lowered from 128 to allow short sequences
        shuffle_buffer: int = 1000,
        seed: int = 42,
        read_chunk_bytes: int = 1024 * 1024,
        hooks: Optional[StreamHooks] = None
    ):
        self.builder = TokenBlockBuilder(
            tokenizer,
            max_length=max_length,
            min_seq_length=min_seq_length
        )
        self.tokenizer = tokenizer  # Keep ref for _make_example
        self.max_length = max_length
        self.hooks = hooks or NullStreamHooks()
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.read_chunk_bytes = read_chunk_bytes

        self.file_info: List[Tuple[Path, int]] = []
        paths = data_path if isinstance(data_path, (list, tuple)) else [data_path]
        for p in paths:
            p = Path(p)
            if p.is_file():
                self.file_info.append((p, p.stat().st_size))
            elif p.is_dir():
                for pat in ("**/*.txt", "**/*.jsonl", "**/*.csv"):
                    for fp in p.glob(pat):
                        self.file_info.append((fp, fp.stat().st_size))

        self.total_bytes = sum(size for _, size in self.file_info)

        print(
            f"[ChunkedTextDataset] {len(self.file_info)} files, "
            f"{self.total_bytes / (1024**2):.1f} MB total"
        )

    def __len__(self):
        return max(1, self.total_bytes // (self.max_length * 2))

    def _make_example(self, token_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Convert token IDs to training example with proper masking."""
        seq_len = len(token_ids)
        if seq_len < self.max_length:
            pad = [self.tokenizer.pad_token_id] * (self.max_length - seq_len)
            mask = [1] * seq_len + [0] * (self.max_length - seq_len)
            token_ids = token_ids + pad
        else:
            token_ids = token_ids[:self.max_length]
            mask = [1] * self.max_length

        # Labels: -100 for padding (ignored in loss)
        labels = [tid if m else -100 for tid, m in zip(token_ids, mask)]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "modality": "text",
            "modality_id": torch.tensor(MODALITY_TEXT, dtype=torch.long),
        }

    def __iter__(self):
        """Stream with global byte-range partitioning and hooks."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        if self.total_bytes == 0:
            print("[ChunkedTextDataset] WARNING: total_bytes=0, no data to iterate")
            return iter(())

        # Diagnostic counters
        lines_read = 0
        tokens_produced = 0
        blocks_yielded = 0

        bytes_per_worker = self.total_bytes // num_workers
        global_start = worker_id * bytes_per_worker
        global_end = self.total_bytes if worker_id == num_workers - 1 else (worker_id + 1) * bytes_per_worker

        rng = random.Random(self.seed + worker_id)
        shuffle_buf = [] if self.shuffle_buffer > 0 else None

        cumulative = 0
        for file_path, file_size in self.file_info:
            file_start = cumulative
            file_end = cumulative + file_size
            cumulative = file_end

            if file_end <= global_start or file_start >= global_end:
                continue

            local_start = max(0, global_start - file_start)
            local_end = min(file_size, global_end - file_start)

            ctx = StreamContext(
                file=file_path,
                worker_id=worker_id,
                byte_start=local_start,
                byte_end=local_end
            )

            for line in iter_lines_in_byte_range(
                file_path,
                start=local_start,
                end=local_end,
                read_chunk_bytes=self.read_chunk_bytes
            ):
                lines_read += 1
                self.hooks.on_line(text=line, ctx=ctx)

                text = self.builder.extract_text(line, suffix=file_path.suffix)
                if not text:
                    continue

                tokens = self.builder.tokenize(text)
                if not tokens:
                    continue

                tokens_produced += len(tokens)
                self.hooks.on_tokens(tokens=tokens, ctx=ctx)

                for block in self.builder.yield_blocks(tokens):
                    blocks_yielded += 1
                    example = self._make_example(block)
                    self.hooks.on_example(example=example, ctx=ctx)

                    if shuffle_buf is not None:
                        shuffle_buf.append(example)
                        if len(shuffle_buf) >= self.shuffle_buffer:
                            rng.shuffle(shuffle_buf)
                            yield from shuffle_buf
                            shuffle_buf.clear()
                    else:
                        yield example

                # Log progress periodically
                if lines_read % 10000 == 0:
                    print(f"[ChunkedTextDataset] worker {worker_id}: {lines_read} lines, {tokens_produced} tokens, {blocks_yielded} blocks")

        if shuffle_buf:
            rng.shuffle(shuffle_buf)
            yield from shuffle_buf

        # Final diagnostic
        print(f"[ChunkedTextDataset] worker {worker_id} done: {lines_read} lines, {tokens_produced} tokens, {blocks_yielded} blocks yielded")


class ImageTextDataset(Dataset):
    """
    Image-text paired dataset.

    Formats supported:
    - JSONL: {'image': 'path/to/image.jpg', 'text': 'caption'}
    - Directory: images/ and captions/ subdirectories

    Args:
        data_path: Path to JSONL or directory
        tokenizer: CognitiveTokenizer
        img_size: Image size (square)
        max_text_length: Max caption length
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        img_size: int = 224,
        max_text_length: int = 128
    ):
        if not VISION_AVAILABLE:
            raise ImportError("PIL and torchvision required for image datasets. Install with: pip install pillow torchvision")

        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load data
        self.examples = []
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Load image-text pairs."""
        path = Path(data_path)

        if path.suffix == '.jsonl':
            self._load_jsonl(path)
        elif path.is_dir():
            self._load_directory(path)
        else:
            raise ValueError(f"Invalid data path: {data_path}")

    def _load_jsonl(self, file_path: Path):
        """Load from JSONL file."""
        base_dir = file_path.parent

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                img_path = base_dir / data['image']
                text = data['text']

                if img_path.exists():
                    self.examples.append({
                        'image_path': str(img_path),
                        'text': text
                    })

    def _load_directory(self, dir_path: Path):
        """Load from directory structure."""
        # Assume images/ and captions.json
        images_dir = dir_path / 'images'
        captions_file = dir_path / 'captions.json'

        if captions_file.exists():
            with open(captions_file, 'r') as f:
                captions = json.load(f)

            for img_name, caption in captions.items():
                img_path = images_dir / img_name
                if img_path.exists():
                    self.examples.append({
                        'image_path': str(img_path),
                        'text': caption
                    })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get image-text pair.

        Returns:
            dict with:
            - image: (3, H, W) normalized image
            - input_ids: (max_text_length,) token IDs
            - attention_mask: (max_text_length,) mask
            - modality: 'image'
        """
        example = self.examples[idx]

        # Load and transform image
        image = Image.open(example['image_path']).convert('RGB')
        image = self.transform(image)

        # Tokenize text
        token_ids = self.tokenizer.encode(
            example['text'],
            max_length=self.max_text_length
        )

        # Pad tokens
        seq_len = len(token_ids)
        if seq_len < self.max_text_length:
            padding = [self.tokenizer.pad_token_id] * (self.max_text_length - seq_len)
            token_ids = token_ids + padding
            attention_mask = [1] * seq_len + [0] * (self.max_text_length - seq_len)
        else:
            attention_mask = [1] * self.max_text_length

        return {
            'image': image,
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'modality': 'image',
            'modality_id': torch.tensor(MODALITY_IMAGE, dtype=torch.long)
        }


class VideoTextDataset(Dataset):
    """
    Video-text paired dataset.

    Supports:
    - Frame directories with captions
    - Video files with annotations

    Args:
        data_path: Path to dataset
        tokenizer: CognitiveTokenizer
        n_frames: Number of frames to sample
        img_size: Frame size
        max_text_length: Max caption length
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        n_frames: int = 8,
        img_size: int = 224,
        max_text_length: int = 128
    ):
        if not VISION_AVAILABLE:
            raise ImportError("PIL and torchvision required for video datasets. Install with: pip install pillow torchvision")

        self.tokenizer = tokenizer
        self.n_frames = n_frames
        self.max_text_length = max_text_length

        # Frame transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load data
        self.examples = []
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Load video-text pairs."""
        path = Path(data_path)

        if path.suffix == '.jsonl':
            self._load_jsonl(path)
        else:
            raise ValueError("Only JSONL format supported for now")

    def _load_jsonl(self, file_path: Path):
        """
        Load from JSONL.

        Format: {'frames_dir': 'path/to/frames/', 'text': 'description'}
        Expects frames named: frame_0000.jpg, frame_0001.jpg, ...
        """
        base_dir = file_path.parent

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                frames_dir = base_dir / data['frames_dir']
                text = data['text']

                if frames_dir.exists():
                    # Get frame paths
                    frame_files = sorted(frames_dir.glob('*.jpg')) + \
                                 sorted(frames_dir.glob('*.png'))

                    if len(frame_files) >= self.n_frames:
                        self.examples.append({
                            'frames_dir': frames_dir,
                            'n_total_frames': len(frame_files),
                            'text': text
                        })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get video-text pair.

        Returns:
            dict with:
            - video: (n_frames, 3, H, W) frames
            - input_ids: (max_text_length,) token IDs
            - attention_mask: (max_text_length,) mask
            - modality: 'video'
        """
        example = self.examples[idx]

        # Sample frames uniformly
        n_total = example['n_total_frames']
        frame_indices = torch.linspace(0, n_total - 1, self.n_frames).long()

        # Load frames
        frames_dir = example['frames_dir']
        frame_files = sorted(frames_dir.glob('*.jpg')) + \
                     sorted(frames_dir.glob('*.png'))

        frames = []
        for i in frame_indices:
            frame = Image.open(frame_files[i]).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)

        video = torch.stack(frames)  # (n_frames, 3, H, W)

        # Tokenize text
        token_ids = self.tokenizer.encode(
            example['text'],
            max_length=self.max_text_length
        )

        # Pad tokens
        seq_len = len(token_ids)
        if seq_len < self.max_text_length:
            padding = [self.tokenizer.pad_token_id] * (self.max_text_length - seq_len)
            token_ids = token_ids + padding
            attention_mask = [1] * seq_len + [0] * (self.max_text_length - seq_len)
        else:
            attention_mask = [1] * self.max_text_length

        return {
            'video': video,
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'modality': 'video',
            'modality_id': torch.tensor(MODALITY_VIDEO, dtype=torch.long)
        }


class MultimodalDataset(Dataset):
    """
    Combined multimodal dataset.

    Merges text, image, and video datasets with sampling weights.

    Args:
        text_dataset: TextDataset instance
        image_dataset: ImageTextDataset instance
        video_dataset: VideoTextDataset instance
        sampling_weights: (text, image, video) sampling probabilities
    """

    def __init__(
        self,
        text_dataset: Optional[TextDataset] = None,
        image_dataset: Optional[ImageTextDataset] = None,
        video_dataset: Optional[VideoTextDataset] = None,
        sampling_weights: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    ):
        self.datasets = []
        self.weights = []
        self.cumulative_sizes = []

        total_size = 0

        if text_dataset is not None:
            self.datasets.append(('text', text_dataset))
            self.weights.append(sampling_weights[0])
            total_size += len(text_dataset)
            self.cumulative_sizes.append(total_size)

        if image_dataset is not None:
            self.datasets.append(('image', image_dataset))
            self.weights.append(sampling_weights[1])
            total_size += len(image_dataset)
            self.cumulative_sizes.append(total_size)

        if video_dataset is not None:
            self.datasets.append(('video', video_dataset))
            self.weights.append(sampling_weights[2])
            total_size += len(video_dataset)
            self.cumulative_sizes.append(total_size)

        self.total_size = total_size

        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get example from one of the datasets.

        Uses idx to deterministically select dataset and example.
        """
        # Find which dataset this idx belongs to
        dataset_idx = 0
        cumulative = 0

        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
            cumulative = size

        # Get example from selected dataset
        modality, dataset = self.datasets[dataset_idx]
        example_idx = idx - cumulative

        return dataset[example_idx]
