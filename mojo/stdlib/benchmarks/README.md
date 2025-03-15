# Mojo standard library benchmarks

This document covers the benchmarks provided for the Mojo
standard library.

## Layout

There is 1-1 correspondence between the directory structure of
the benchmarks and their source counterpart.  For example,
consider `collections/bench_dict.mojo` and its source counterpart:
`collections/dict.mojo`.  This organization makes it easy to stay
organized.

Benchmark files should be prefixed with `bench_` in the filename.
This is helpful for consistency, but also is recognized by tooling
internally.

## How to run the benchmarks

Running e.g. `magic run benchmarks ./stdlib/benchmarks/collections` runs every
file in the directory sequentially using the build based on the current branch.
It can also run one file when specified.

## How to write effective benchmarks

All of the benchmarks use the `benchmark` module.  `Bench` objects are built
on top of the `benchmark` module.  You can also use `BenchConfig` to configure
`Bench`. For the most part, you can copy-paste from existing
benchmarks to get started.

Note that the `benchmark` package isn't open source yet and we do not currently
have a mechanism for generating nightly API docs for closed source packages.
So, we manually provide relatively up-to-date docs for these [here](../../docs/bencher/).
In the future, we hope to open source the `benchmark` package and and also generate
nightly API docs.  This is definitely a rough edge, but bear with us!  We eagerly
wanted to get these benchmarks out to the public even though we fully understand
the experience is not perfect right now.

## Benchmarks in CI

Currently, there is no short-term plans for adding these benchmarks with regression
detection and such in the public Mojo CI.  We're working hard to improve the processes
for this internally first before we commit to doing this in the external repo.

## Other reading

Check out our [blog post](https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking)
for more info on writing benchmarks.
