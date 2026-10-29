from __future__ import annotations

from typing import Any, Protocol

from src.inputs import RGBImage


class RestorationError(RuntimeError):
    pass


class Method(Protocol):
    name: str

    def restore(self, image: RGBImage) -> RGBImage: ...

    def metadata(self) -> dict[str, Any]: ...

    def synchronize(self) -> None: ...


class Identity:
    name = 'identity'

    def restore(self, image: RGBImage) -> RGBImage:
        return image.copy()

    def metadata(self) -> dict[str, Any]:
        return {'device': 'cpu', 'policy': 'decoded RGB identity', 'startup_ms': 0.0}

    def synchronize(self) -> None:
        return None


class Median(Identity):
    name = 'median3'

    def restore(self, image: RGBImage) -> RGBImage:
        from scipy.ndimage import median_filter
        return median_filter(image, size=(3, 3, 1), mode='reflect')  # type: ignore[no-any-return]

    def metadata(self) -> dict[str, Any]:
        return {'device': 'cpu', 'window': [3, 3, 1], 'border': 'reflect', 'startup_ms': 0.0}
