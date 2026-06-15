from __future__ import annotations


def test_e2e_package_importable() -> None:
    """Verify e2e test package imports resolve."""
    from kuchikae.pipeline import create_pipeline, KuchikaePipeline
    from kuchikae.ui import create_app
    assert create_pipeline is not None
    assert KuchikaePipeline is not None
    assert create_app is not None
