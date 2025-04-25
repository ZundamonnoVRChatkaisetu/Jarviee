"""
Multimodal AI Adapter Package for Jarviee System.

This package implements the adapter for integrating multimodal AI technologies
with the Jarviee system. It provides a bridge between the LLM core and
multimodal processing capabilities, enabling language and other modalities
(images, audio, sensors) to be integrated for enhanced understanding and interaction.
"""

from .adapter import MultimodalAdapter

__all__ = ["MultimodalAdapter"]
