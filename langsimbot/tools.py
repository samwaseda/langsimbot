import numpy as np
from langchain.agents import tool


@tool
def get_elastic_constants(chemical_symbol: str) -> dict:
    """
    Return elastic constants based on the chemical symbol

    Args:
        chemical_symbol (str): Chemical symbol (e.g. "Au", "Fe" etc.)

    Returns:
        (dict): Elastic constants
    """
    C_11 = np.mean([ord(letter) for letter in chemical_symbol])
    C_12 = C_11 / 2
    C_44 = 0.5 * (C_11 - C_12)
    return {"C_11": C_11, "C_12": C_12, "C_44": C_44}
