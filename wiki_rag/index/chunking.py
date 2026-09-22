#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Section chunking strategies applied at index time.

All limits are measured in UTF-8 **bytes**, not characters, because the
the ``text`` storage limit is byte-based (5000 UTF-8 bytes per chunk).
Splits never break a multi-byte codepoint and prefer whitespace boundaries.
"""

import logging

from dataclasses import dataclass

logger = logging.getLogger(__name__)

STRATEGIES = ("none", "fixed", "paragraph")


@dataclass(frozen=True)
class Chunk:
    """One chunk produced by :func:`split_section`.

    Attributes:
        text: Overlap-free body text stored in the vector store.  Joining
            ``chunk.text + chunk.separator`` across all chunks in document
            order reproduces the original (stripped) section body exactly.
        embed_text: Body text fed to the embedding model.  Identical to
            *text* for the first chunk and for strategies that do not use
            overlap; otherwise prefixed with the trailing bytes of the
            previous chunk's body so that concepts spanning a chunk boundary
            are captured in both dense vectors.
        separator: Whitespace that originally separated the end of this
            chunk's body from the start of the next one in the source section
            (``""`` for the last chunk, which has no successor).

    """

    text: str
    embed_text: str
    separator: str


def split_section(
        text: str,
        *,
        strategy: str,
        max_bytes: int,
        overlap_bytes: int = 0,
) -> list[Chunk]:
    """Split a section body into chunks according to the given strategy.

    Args:
        text: Tidied section body text.
        strategy: One of ``"none"`` (trim at limit, historical behaviour),
            ``"fixed"`` (hard split on whitespace boundaries; overlap is
            applied to :attr:`Chunk.embed_text` only — the stored
            :attr:`Chunk.text` is overlap-free) or ``"paragraph"`` (greedy
            paragraph packing, fixed fallback for oversized paragraphs, no
            overlap).
        max_bytes: Maximum UTF-8 byte length of every returned chunk's
            stored :attr:`Chunk.text`.
        overlap_bytes: Approximate number of trailing bytes from the previous
            chunk's body prepended to the next chunk's :attr:`Chunk.embed_text`
            (``"fixed"`` strategy only; ignored for all other strategies).

    Returns:
        List of :class:`Chunk` objects in document order.  Empty input
        yields an empty list.  Joining ``chunk.text + chunk.separator``
        across all chunks reproduces the original (stripped) section body.

    Raises:
        ValueError: If *strategy* is not one of :data:`STRATEGIES`.

    """
    if strategy not in STRATEGIES:
        msg = f"Unknown chunking strategy: {strategy!r}. Valid: {STRATEGIES}"
        raise ValueError(msg)

    text = text.strip()
    if not text:
        return []

    if len(text.encode("utf-8")) <= max_bytes:
        return [Chunk(text=text, embed_text=text, separator="")]

    if strategy == "none":
        trimmed = _trim_to_bytes(text, max_bytes)
        return [Chunk(text=trimmed, embed_text=trimmed, separator="")]
    if strategy == "fixed":
        return _split_fixed(text, max_bytes, overlap_bytes)
    return _split_paragraph(text, max_bytes)


def _trim_to_bytes(text: str, max_bytes: int) -> str:
    """Trim *text* to at most *max_bytes* UTF-8 bytes on a codepoint boundary."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore").strip()


def _normalise_separator(ws: str) -> str:
    r"""Normalise a raw whitespace run to a canonical separator string.

    Args:
        ws: Raw whitespace characters captured between two adjacent chunk
            bodies (may include spaces, tabs, and newlines).

    Returns:
        ``""`` when *ws* is empty (mid-word cut — chunks rejoin seamlessly),
        ``"\n\n"`` when two or more newlines are present,
        ``"\n"`` for a single newline, and ``" "`` for spaces/tabs only.

    """
    if not ws:
        return ""
    if ws.count("\n") >= 2:
        return "\n\n"
    if "\n" in ws:
        return "\n"
    return " "


def _split_fixed(text: str, max_bytes: int, overlap_bytes: int) -> list[Chunk]:
    """Split on the nearest whitespace at or before *max_bytes*, with embed-only overlap.

    Overlap is captured only in :attr:`Chunk.embed_text`; :attr:`Chunk.text`
    is the overlap-free body so that section reconstruction is clean.
    Joining ``chunk.text + chunk.separator`` across all returned chunks
    reproduces *text*.

    Args:
        text: Section body (already stripped).
        max_bytes: Maximum byte length for each stored body.
        overlap_bytes: Bytes from the previous core to prepend to embed_text.

    Returns:
        List of :class:`Chunk` objects.

    """
    # Clamp the overlap so every iteration always consumes more new text than
    # it repeats, guaranteeing termination even with extreme settings.
    overlap_bytes = min(overlap_bytes, max_bytes // 4)

    # First pass: collect overlap-free cores and the raw separator that
    # originally separated consecutive cores in the source text.
    cores: list[str] = []
    separators: list[str] = []  # len(separators) == len(cores) - 1

    remaining = text
    while remaining:
        if len(remaining.encode("utf-8")) <= max_bytes:
            cores.append(remaining)
            break
        head = _trim_to_bytes(remaining, max_bytes)
        # Prefer a whitespace boundary so words are not cut in half, but only
        # when one exists in the tail half of the head (avoids degenerate
        # tiny chunks on whitespace-poor text).
        cut = head.rfind(" ", len(head) // 2)
        if cut > 0:
            head = head[:cut]
        core = head.rstrip()
        cores.append(core)

        # Capture the whitespace run immediately following the core in the
        # original string; that run is the natural separator between this
        # chunk and the next.
        after = remaining[len(head):]
        i = 0
        while i < len(after) and after[i] in " \t\n\r":
            i += 1
        separators.append(_normalise_separator(after[:i]))
        remaining = after[i:]

    # Second pass: build Chunk objects.  embed_text for chunk i > 0 prepends
    # the trailing bytes of core i-1 so that concepts spanning the boundary
    # are captured in both dense vectors (the overlap effect), while text
    # stores only the overlap-free body for clean section reconstruction.
    chunks: list[Chunk] = []
    for i, core in enumerate(cores):
        is_last = i == len(cores) - 1
        separator = "" if is_last else separators[i]

        if i == 0 or overlap_bytes == 0:
            embed_text = core
        else:
            prev_core = cores[i - 1]
            overlap_raw = _trim_to_bytes(prev_core[-(overlap_bytes * 2):], overlap_bytes)
            # Start the overlap prefix on a word boundary.
            overlap_pfx = overlap_raw.split(" ", 1)[-1] if " " in overlap_raw else overlap_raw
            embed_text = f"{overlap_pfx} {core}"

        if core:
            chunks.append(Chunk(text=core, embed_text=embed_text, separator=separator))

    return chunks


def _split_paragraph(text: str, max_bytes: int) -> list[Chunk]:
    r"""Pack whole paragraphs greedily; oversized paragraphs use fixed split.

    Adjacent chunks are joined by ``"\n\n"`` (the original paragraph break).
    Oversized paragraphs are split by :func:`_split_fixed` with
    ``overlap_bytes=0`` and their intra-paragraph separators are preserved;
    the last sub-chunk of each such paragraph connects to the following
    content with ``"\n\n"``.

    Args:
        text: Section body (already stripped).
        max_bytes: Maximum byte length for each stored body.

    Returns:
        List of :class:`Chunk` objects.  No overlap is applied.

    """
    separator = "\n\n"
    separator_bytes = len(separator.encode("utf-8"))

    all_chunks: list[Chunk] = []
    current: list[str] = []
    current_bytes = 0

    for paragraph in (p.strip() for p in text.split(separator)):
        if not paragraph:
            continue
        paragraph_bytes = len(paragraph.encode("utf-8"))
        if paragraph_bytes > max_bytes:
            # Flush whatever is packed so far, then fixed-split the oversized
            # paragraph.  Sub-chunks carry their own intra-paragraph separators;
            # the last sub-chunk gets a provisional "\n\n" to connect it to
            # whatever follows — corrected to "" at the end if it turns out to
            # be the last chunk overall.
            if current:
                body = separator.join(current)
                all_chunks.append(Chunk(text=body, embed_text=body, separator="\n\n"))
                current, current_bytes = [], 0
            sub_chunks = _split_fixed(paragraph, max_bytes, overlap_bytes=0)
            for j, sc in enumerate(sub_chunks):
                is_last_sub = j == len(sub_chunks) - 1
                sep = "\n\n" if is_last_sub else sc.separator
                all_chunks.append(Chunk(text=sc.text, embed_text=sc.embed_text, separator=sep))
            continue
        if current and current_bytes + separator_bytes + paragraph_bytes > max_bytes:
            body = separator.join(current)
            all_chunks.append(Chunk(text=body, embed_text=body, separator="\n\n"))
            current, current_bytes = [], 0
        current_bytes += (separator_bytes if current else 0) + paragraph_bytes
        current.append(paragraph)

    if current:
        body = separator.join(current)
        all_chunks.append(Chunk(text=body, embed_text=body, separator="\n\n"))

    # Correct the last chunk's separator: nothing follows it.
    if all_chunks:
        last = all_chunks[-1]
        all_chunks[-1] = Chunk(text=last.text, embed_text=last.embed_text, separator="")

    return all_chunks
