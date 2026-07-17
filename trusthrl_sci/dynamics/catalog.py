from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Cytokine(IntEnum):
    EOTAXIN = 0
    G_CSF = 1
    GM_CSF = 2
    IFN_GAMMA = 3
    IL_1_ALPHA = 4
    IL_1_BETA = 5
    IL_2 = 6
    IL_4 = 7
    IL_5 = 8
    IL_6 = 9
    IL_7 = 10
    IL_10 = 11
    IL_12_P70 = 12
    IL_13 = 13
    IL_15 = 14
    IL_17A = 15
    IL_18 = 16
    IP_10 = 17
    GRO_KC = 18
    LIX = 19
    MCP_1 = 20
    MIP_1_ALPHA = 21
    MIP_2 = 22
    RANTES = 23
    TNF_ALPHA = 24
    VEGF = 25
    TIMP_1 = 26
    LEPTIN = 27


@dataclass(frozen=True)
class CytokineDescriptor:
    index: int
    symbol: str
    family: str
    pro_inflammatory: bool
    anti_inflammatory: bool
    remodeling: bool


CATALOG = (
    CytokineDescriptor(0, "Eotaxin", "chemokine", False, False, False),
    CytokineDescriptor(1, "G-CSF", "growth_factor", True, False, False),
    CytokineDescriptor(2, "GM-CSF", "growth_factor", True, False, False),
    CytokineDescriptor(3, "IFN-gamma", "interferon", True, False, False),
    CytokineDescriptor(4, "IL-1alpha", "interleukin", True, False, False),
    CytokineDescriptor(5, "IL-1beta", "interleukin", True, False, False),
    CytokineDescriptor(6, "IL-2", "interleukin", True, False, False),
    CytokineDescriptor(7, "IL-4", "interleukin", False, True, False),
    CytokineDescriptor(8, "IL-5", "interleukin", False, True, False),
    CytokineDescriptor(9, "IL-6", "interleukin", True, False, False),
    CytokineDescriptor(10, "IL-7", "interleukin", False, False, False),
    CytokineDescriptor(11, "IL-10", "interleukin", False, True, False),
    CytokineDescriptor(12, "IL-12p70", "interleukin", True, False, False),
    CytokineDescriptor(13, "IL-13", "interleukin", False, True, False),
    CytokineDescriptor(14, "IL-15", "interleukin", True, False, False),
    CytokineDescriptor(15, "IL-17A", "interleukin", True, False, False),
    CytokineDescriptor(16, "IL-18", "interleukin", True, False, False),
    CytokineDescriptor(17, "IP-10", "chemokine", True, False, False),
    CytokineDescriptor(18, "GRO-KC", "chemokine", True, False, False),
    CytokineDescriptor(19, "LIX", "chemokine", True, False, False),
    CytokineDescriptor(20, "MCP-1", "chemokine", True, False, False),
    CytokineDescriptor(21, "MIP-1alpha", "chemokine", True, False, False),
    CytokineDescriptor(22, "MIP-2", "chemokine", True, False, False),
    CytokineDescriptor(23, "RANTES", "chemokine", True, False, False),
    CytokineDescriptor(24, "TNF-alpha", "tumor_necrosis_factor", True, False, False),
    CytokineDescriptor(25, "VEGF", "growth_factor", False, False, True),
    CytokineDescriptor(26, "TIMP-1", "matrix_regulator", False, False, True),
    CytokineDescriptor(27, "Leptin", "adipokine", False, False, False),
)


def pro_inflammatory_indices() -> tuple[int, ...]:
    return tuple(item.index for item in CATALOG if item.pro_inflammatory)


def anti_inflammatory_indices() -> tuple[int, ...]:
    return tuple(item.index for item in CATALOG if item.anti_inflammatory)


def remodeling_indices() -> tuple[int, ...]:
    return tuple(item.index for item in CATALOG if item.remodeling)


def symbols() -> tuple[str, ...]:
    return tuple(item.symbol for item in CATALOG)


def resolve_symbol(symbol: str) -> int:
    normalized = symbol.lower().replace("_", "-")
    for descriptor in CATALOG:
        if descriptor.symbol.lower() == normalized:
            return descriptor.index
    raise KeyError(symbol)
