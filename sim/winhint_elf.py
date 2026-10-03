"""
winhint_elf.py - static scan of a RISC-V ELF for WinHint hint instructions.

Stdlib only: imported by sim/se.py (inside gem5's embedded Python) and by the
host-side tools (sim/run_lengths.py, run_experiments.py).

docs/interfaces.md §2: a hint is ``ori x0, x0, IMM`` with IMM[11] = 0 and
IMM[4:0] = 21 (setwin, W = IMM[10:5] * 8) or 23 (region, id = IMM[10:5]).
The raw word is ``0x00006013 | (IMM << 20)``.

Every 2-byte aligned position of every executable section is checked (RVC
code is 2-byte aligned). A match that is not an instruction start (e.g. the
upper half of another instruction) is harmless: the result is only used to
recognise the PCs that *retire*, and a 32-bit instruction that retires at such
a PC is exactly that hint word.
"""

from __future__ import annotations

import struct

#: IMM[4:0] tag of setwin(W)
SETWIN_TAG = 0x15  # 0b10101
#: IMM[4:0] tag of region(id)
REGION_TAG = 0x17  # 0b10111
_LOW20 = 0x06013   # opcode OP-IMM, rd = x0, funct3 = ORI, rs1 = x0
#: ELF section flag: section contains executable instructions
SHF_EXECINSTR = 0x4


def decode_hint(word: int):
    """Decode a 32-bit instruction word as a WinHint hint.

    Args:
        word: Raw little-endian instruction word.

    Returns:
        hint (tuple[int, int] | None): ``(kind, payload)`` with kind 1 = setwin
            (payload = IMM[10:5], W = payload * 8) or 2 = region (payload =
            region id); None if the word is not a hint.
    """
    if word & 0xFFFFF != _LOW20:
        return None
    imm = (word >> 20) & 0xFFF
    if imm & 0x800:
        return None
    tag, payload = imm & 0x1F, (imm >> 5) & 0x3F
    if tag == SETWIN_TAG:
        return 1, payload
    if tag == REGION_TAG:
        return 2, payload
    return None


def _exec_sections(data: bytes):
    """Yield the executable PROGBITS sections of an ELF image.

    Args:
        data: Whole ELF file contents.

    Yields:
        section (tuple[int, bytes]): ``(sh_addr, section_bytes)`` for each non-empty section with
            ``SHF_EXECINSTR``.

    Raises:
        ValueError: If ``data`` is not an ELF file or not little-endian ELF64.
    """
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("only little-endian ELF64 is supported")
    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from(
            "<IQQQQ", data, off + 4)
        if sh_type == 1 and sh_flags & SHF_EXECINSTR and sh_size:  # PROGBITS
            yield sh_addr, data[sh_offset:sh_offset + sh_size]


def scan_hints(path) -> dict:
    """Return every hint in the executable sections of an ELF file.

    Args:
        path (str | os.PathLike): Path to a little-endian ELF64 (RISC-V) binary.

    Returns:
        ``{"setwin": {pc: W}, "region": {pc: id}}`` where W is the requested
        window size in entries (IMM[10:5] * 8).

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is not a little-endian ELF64.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    out = {"setwin": {}, "region": {}}
    for addr, sec in _exec_sections(data):
        i = sec.find(b"\x13\x60")
        while i >= 0:
            if i % 2 == 0 and i + 4 <= len(sec):
                h = decode_hint(int.from_bytes(sec[i:i + 4], "little"))
                if h is not None:
                    kind, payload = h
                    if kind == 1:
                        out["setwin"][addr + i] = payload * 8
                    else:
                        out["region"][addr + i] = payload
            i = sec.find(b"\x13\x60", i + 1)
    return out


def config_for_setwin(w: int, rob: list[int]) -> int:
    """Map a setwin window size to a window-table configuration index.

    docs/interfaces.md §3: smallest config with ROB >= W; W = 0 or too large
    -> the largest. Mirrors WindowTable::configForSetwin (policy.hh): the table
    is ascending, the first fitting index wins, largest = last index.

    Args:
        w: Requested window size W from the setwin hint.
        rob: Ascending per-config ROB sizes (machine JSON ``window.rob``).

    Returns:
        The configuration index.
    """
    largest = len(rob) - 1
    if w <= 0:
        return largest
    for i, r in enumerate(rob):
        if r >= w:
            return i
    return largest
