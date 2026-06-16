"""Audio noise reduction utilities."""

from __future__ import annotations

import logging
import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("kuchikae.audio.denoise")


def reduce_noise(
    audio_path: str,
    output_path: str | None = None,
    stationary: bool = False,
    prop_decrease: float = 0.8,
    freq_mask_smooth_hz: int = 500,
    time_mask_smooth_ms: int = 50,
    thresh_n_mult_nonstationary: int = 2,
    sigmoid_slope_nonstationary: int = 10,
    n_std_thresh_stationary: float = 1.5,
) -> str:
    """Reduce noise in audio file using spectral gating.

    Args:
        audio_path: Path to input audio file.
        output_path: Path to output audio file. If None, creates temp file.
        stationary: If True, use stationary noise reduction (assumes constant noise).
        prop_decrease: Proportion to reduce noise (0.0 to 1.0). 1.0 = full reduction.
        freq_mask_smooth_hz: Frequency smoothing in Hz.
        time_mask_smooth_ms: Time smoothing in milliseconds.
        thresh_n_mult_nonstationary: Multiplier for non-stationary threshold.
        sigmoid_slope_nonstationary: Sigmoid slope for non-stationary reduction.
        n_std_thresh_stationary: Number of standard deviations for stationary threshold.

    Returns:
        Path to denoised audio file.
    """
    try:
        import noisereduce as nr
    except ImportError:
        logger.warning("noisereduce not available, skipping noise reduction")
        return audio_path

    try:
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Handle stereo to mono conversion
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
            sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
        )
        
        # Save output
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav", prefix="denoised_")
        
        sf.write(output_path, reduced, sr)
        logger.info("denoise: %s -> %s", audio_path, output_path)
        return output_path
        
    except Exception as e:
        logger.warning("denoise failed: %s, returning original", e)
        return audio_path


def reduce_noise_simple(
    audio_path: str,
    output_path: str | None = None,
    prop_decrease: float = 0.75,
) -> str:
    """Simplified noise reduction with sensible defaults.

    Args:
        audio_path: Path to input audio file.
        output_path: Path to output audio file. If None, creates temp file.
        prop_decrease: Proportion to reduce noise (0.0 to 1.0).

    Returns:
        Path to denoised audio file.
    """
    return reduce_noise(
        audio_path=audio_path,
        output_path=output_path,
        stationary=False,
        prop_decrease=prop_decrease,
    )
