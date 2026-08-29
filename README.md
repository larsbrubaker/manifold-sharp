# manifold-sharp

Pure C# port of the [Manifold](https://github.com/elalish/manifold) 3D geometry
library, ported from [manifold-rust](https://github.com/larsbrubaker/manifold-rust)
(itself an exact-match port of Manifold C++ v3.5.x).

Unlike the `ManifoldRust` .NET binding (P/Invoke over a Rust cdylib), this is
fully managed code: no native library to build, load, or link — it runs anywhere
.NET runs, including `browser-wasm`, with no `NativeFileReference` or emcc step.

**Exactness bar:** identical results on identical inputs versus manifold-rust
(and therefore versus the C++ reference). Ports are verified test-for-test
against the Rust suite, and cross-checked numerically against the native
library through the existing `ManifoldRust` binding.

See [docs/PORTING_PLAN.md](docs/PORTING_PLAN.md) for the roadmap.

## License

[Apache-2.0](LICENSE), same as Manifold and manifold-rust.
