#!/usr/bin/env python3

from typing import List, Optional, Set


class BinaryNinjaFunctionSignatureGenerator:
    """Best-effort Binary Ninja implementation of Sigga-style masking."""

    def __init__(self, binary_view):
        self.binary_view = binary_view

    def generate(self, function) -> Optional[str]:
        if function is None or self.binary_view is None:
            return None

        tokens: List[str] = []
        seen_addresses: Set[int] = set()
        for basic_block in getattr(function, "basic_blocks", []) or []:
            for line in getattr(basic_block, "disassembly_text", []) or []:
                address = int(getattr(line, "address", 0) or 0)
                if not address or address in seen_addresses:
                    continue
                seen_addresses.add(address)

                try:
                    length = int(self.binary_view.get_instruction_length(address) or 0)
                except Exception:
                    length = 0
                if length <= 0:
                    continue

                data = self.binary_view.read(address, length)
                if not data:
                    continue

                masked = self._mask_instruction(str(line).strip(), bytes(data))
                tokens.extend(masked)
                if len(tokens) >= 64:
                    break
            if len(tokens) >= 64:
                break

        trimmed = self._trim_trailing_wildcards(tokens)
        return " ".join(trimmed) if trimmed else None

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
        return any(marker in operand_text for marker in (
            "rip",
            " ptr ",
            "[",
            "got",
            "plt",
            "extern",
            "off_",
            "sub_",
            "data_",
            "0x",
        ))

    @staticmethod
    def _trim_trailing_wildcards(tokens: List[str]) -> List[str]:
        trimmed = list(tokens)
        while trimmed and trimmed[-1] == "?":
            trimmed.pop()
        return trimmed
