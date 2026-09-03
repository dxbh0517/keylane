"""Which of a runtime's advertised devices this machine can actually use.

``RuntimeInfo.devices`` is a static list of what the *stack* supports. It is
not a list of what is present, and on at least one common configuration the gap
is dangerous rather than merely untidy: OpenVINO enumerates any OpenCL device it
can see, so on a laptop with a discrete NVIDIA card ``Core().available_devices``
returns ``['CPU', 'GPU', 'NPU']`` where ``GPU`` is the NVIDIA part. Offering
that in Settings invites a compile that cannot succeed, and the user gets a
crash instead of a reason.

So a device is offered only when it is present *and* it belongs to a vendor the
runtime can compile for. Anything present but unusable is still reported, with
the reason, so Settings can grey it out and say why rather than hiding hardware
the user knows they have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# OpenVINO's GPU plugin compiles for Intel graphics. A name from any other
# vendor means OpenVINO enumerated a device it cannot target.
_FOREIGN_VENDORS = ("nvidia", "geforce", "quadro", "radeon", "amd ", "advanced micro")


@dataclass(frozen=True)
class DeviceOption:
    """One device a runtime could be pointed at."""

    id: str
    label: str
    usable: bool
    reason: str = ""


def _openvino_devices() -> dict[str, str]:
    """``{device id: full name}`` as OpenVINO sees the machine, or empty."""
    try:
        import openvino as ov  # noqa: PLC0415

        core = ov.Core()
        found: dict[str, str] = {}
        for device in core.available_devices:
            # A multi-GPU box reports GPU.0, GPU.1 …; the bare id is the first.
            key = device.split(".")[0].upper()
            try:
                name = str(core.get_property(device, "FULL_DEVICE_NAME"))
            except Exception:  # noqa: BLE001
                name = device
            found.setdefault(key, name)
        return found
    except Exception:  # noqa: BLE001
        logger.debug("could not enumerate OpenVINO devices", exc_info=True)
        return {}


def _is_foreign(full_name: str) -> bool:
    lowered = full_name.lower()
    return any(vendor in lowered for vendor in _FOREIGN_VENDORS)


def device_options(advertised: tuple[str, ...]) -> list[DeviceOption]:
    """Annotate a runtime's advertised devices with what is really here.

    ``AUTO`` is never a physical device — it means "leave the model's own
    provider alone" — so it is always offered and never probed.
    """
    present = _openvino_devices()
    options: list[DeviceOption] = []
    for device in advertised:
        key = device.upper()
        if key == "AUTO":
            options.append(DeviceOption(key, "Auto (as exported)", True))
            continue
        if not present:
            # Nothing to check against — do not invent an objection.
            options.append(DeviceOption(key, key, True))
            continue
        full_name = present.get(key)
        if full_name is None:
            options.append(DeviceOption(key, key, False, "not present on this machine"))
        elif _is_foreign(full_name):
            options.append(
                DeviceOption(
                    key,
                    f"{key} — {full_name}",
                    False,
                    f"{full_name} is not an Intel device; OpenVINO cannot compile for it",
                )
            )
        else:
            options.append(DeviceOption(key, f"{key} — {full_name}", True))
    return options


def usable_device_ids(advertised: tuple[str, ...]) -> list[str]:
    return [option.id for option in device_options(advertised) if option.usable]
