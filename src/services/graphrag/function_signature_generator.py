#!/usr/bin/env python3

import re
from typing import List, Optional, Set


class BinaryNinjaFunctionSignatureGenerator:
    """Canonical masked-prefix signature generator shared across plugins."""

    MAX_SIGNATURE_BYTES = 64

    def __init__(self, binary_view):
        self.binary_view = binary_view

    def generate(self, function) -> Optional[str]:
        if function is None or self.binary_view is None:
            return None

        tokens: List[str] = []
        for address in self._iter_instruction_addresses(function):
            try:
                length = int(self.binary_view.get_instruction_length(address) or 0)
            except Exception:
                length = 0
            if length <= 0:
                continue

            data = self.binary_view.read(address, length)
            if not data:
                continue

            try:
                text = self.binary_view.get_disassembly(address) or ""
            except Exception:
                text = ""

            masked = self._mask_instruction(text, bytes(data))
            for token in masked:
                tokens.append(token)
                if len(tokens) >= self.MAX_SIGNATURE_BYTES:
                    trimmed = self._trim_trailing_wildcards(tokens)
                    return " ".join(trimmed) if trimmed else None

        trimmed = self._trim_trailing_wildcards(tokens)
        return " ".join(trimmed) if trimmed else None

    def _iter_instruction_addresses(self, function) -> List[int]:
        seen_addresses: Set[int] = set()
        addresses: List[int] = []
        for basic_block in getattr(function, "basic_blocks", []) or []:
            for line in getattr(basic_block, "disassembly_text", []) or []:
                address = int(getattr(line, "address", 0) or 0)
                if not address or address in seen_addresses:
                    continue
                seen_addresses.add(address)
                addresses.append(address)
        addresses.sort()
        return addresses

    def _mask_instruction(self, text: str, data: bytes) -> List[str]:
        tokens = [f"{byte:02X}" for byte in data]
        if not tokens:
            return tokens

        parts = text.split(None, 1)
        mnemonic = parts[0].lower() if parts else ""
        operand_text = parts[1].lower() if len(parts) > 1 else ""

        if self._is_branch_like(mnemonic):
            for index in range(1, len(tokens)):
                tokens[index] = "?"
            return tokens

        if self._should_mask_operands(operand_text) and len(tokens) > 1:
            start = max(1, len(tokens) - min(4, len(tokens) - 1))
            for index in range(start, len(tokens)):
                tokens[index] = "?"

        return tokens

    @staticmethod
    def _is_branch_like(mnemonic: str) -> bool:
        return (
            mnemonic.startswith("j")
            or mnemonic.startswith("call")
            or mnemonic.startswith("b")
            or mnemonic in {"loop", "loopne", "loope"}
        )

    @staticmethod
    def _should_mask_operands(operand_text: str) -> bool:
        markers = (
            "rip",
            " ptr ",
            "[",
            "got",
            "plt",
            "extern",
            "extrn",
            "offset",
            "off_",
            "sub_",
            "loc_",
            "data_",
            "cs:",
            "ds:",
            "0x",
        )
        if any(marker in operand_text for marker in markers):
            return True
        return re.search(r"\b[0-9a-f]+h\b", operand_text) is not None

    @staticmethod
    def _trim_trailing_wildcards(tokens: List[str]) -> List[str]:
        trimmed = list(tokens)
        while trimmed and trimmed[-1] == "?":
            trimmed.pop()
        return trimmed
