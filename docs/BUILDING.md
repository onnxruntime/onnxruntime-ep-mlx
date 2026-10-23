# Build from source

For normal Python use, install the wheel as described in the [README](../README.md).
These instructions are for building the native plugin on macOS 14+ / Apple Silicon.

## Native plugin

Install a current stable Rust toolchain (edition 2024), CMake 3.25+, a C++20
compiler with the macOS SDK and Metal build tools, `curl`, and `unzip`.
You also need the ONNX Runtime 1.31 headers. From the repository root:

```sh
rust/scripts/setup_mlx.sh "$PWD/.deps/mlx-0.32.4"
export MLX_PREFIX="$PWD/.deps/mlx-0.32.4"
export MLXC_PREFIX="$MLX_PREFIX"

curl -fsSL \
  https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-osx-arm64-1.31.0.tgz \
  | tar xz
export ORT_HOME="$PWD/onnxruntime-osx-arm64-1.31.0"

cargo build --manifest-path rust/Cargo.toml --release
# Output: rust/target/release/libonnxruntime_mlx_ep.dylib
```

The setup script builds pinned MLX 0.32.4 (`36969c09`) and mlx-c (`57731b00`).
Both are required dependencies; mlx-c's v0.7.0 tag targets MLX 0.32.2 instead.
Use an empty prefix or one already provisioned with the same pins.

CI caches only the installed prefix, shared by the independent Clippy, build/test,
and wheel jobs. Exact cache keys include the runner OS/architecture, MLX version,
both source commits, compiler/Xcode/SDK/macOS/CMake identity, deployment target and
build flags, and the setup script/action content hash. There are no fallback restore
keys across these boundaries. Every restore still runs the setup script's stamp,
header and runtime-file checks; an invalid nonempty prefix fails explicitly.
Concurrent cold-cache jobs may each build the runtime; later matching runs reuse it.

Alternatively, set `ORT_INCLUDE_DIR` to the 1.31 include directory; it takes
precedence over `ORT_HOME/include`. Each MLX prefix must contain `include/` and
`lib/`. Keep those libraries available at runtime: the standalone dylib links
to MLX and mlx-c in those prefixes.

The crate generates ORT and mlx-c bindings with `bindgen`. It does **not** link
`libonnxruntime`: ORT calls go through the `OrtApi` function-pointer table passed
to `CreateEpFactories`.

## Python wheel

With the dependencies and environment above configured:

```sh
python -m pip install build
MACOSX_DEPLOYMENT_TARGET=14.0 python -m build --wheel ./python -o dist
python -m pip install dist/onnxruntime_ep_mlx-*.whl
```

The Hatchling hook builds the Rust dylib, bundles and relinks MLX and mlx-c with
`delocate`, and includes `mlx.metallib` and an optional `libjaccl.dylib` from
`MLX_PREFIX/lib`. The wheel has no compiled Python extension, but still requires
a compatible ONNX Runtime installation for the interpreter.

See [testing](../tests/README.md) for validation commands and
[integration](INTEGRATION.md) for loading the plugin outside Python.
