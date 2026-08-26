from dataclasses import dataclass


def configclass(cls):
    """rtnav config convention: a frozen dataclass."""
    return dataclass(frozen=True)(cls)
