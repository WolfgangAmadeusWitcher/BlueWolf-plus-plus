# Contributing

Thanks for your interest in BW++.

## How to contribute
- Open an issue with a clear description and repro steps.
- Propose changes via pull requests.
- Keep changes focused and small.

## Development setup
- Build compiler: `make -C compiler`
- Run compiler (stub): `compiler/bwppc examples/matmul.bwpp out.metal`

## Style
- C code: C11, `clang -Wall -Wextra -Werror`
- Docs: concise, ASCII

## Performance claims
Benchmark results must include:
- hardware and OS
- compiler version
- input sizes
- wall time and throughput

## Code of Conduct
By participating, you agree to `CODE_OF_CONDUCT.md`.
