# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Custom Click Options used in pipelines"""

import click


class DevicesOptionType(click.ParamType):
    name = "devices"

    def convert(self, value, param, ctx):
        # Returns an empty list of devices if nothing is provided.
        if not value:
            return []
        elif value == "cpu" or value == "gpu":
            return value
        try:
            # Also account for non gpu prefixed ids (backwards compatibility)
            results = [
                int(i.replace("gpu-", "")) if i.startswith("gpu-") else int(i)
                for i in value.split(",")
            ]
            return results
        except ValueError:
            self.fail(
                (
                    f"{value!r} is not a valid device list - must be a"
                    " comma-separated list of gpu-<N> ids."
                ),
                param,
                ctx,
            )
