# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Entrypoint for generalized click cli."""

import click


class ModelGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Command not supported: {cmd_name}\nSupported commands:"
            f" {supported}"
        )


@click.command(cls=ModelGroup)
def run_cli():
    pass
