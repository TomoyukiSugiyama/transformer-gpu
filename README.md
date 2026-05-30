# transformer-gpu

Rust で書かれた Transformer (decoder-only) 言語モデルの学習・推論実装。
外部 ML フレームワークに依存せず、 行列演算から自前で**GPU**実装している学習用プロジェクト。

[Transformer (decoder-only) 言語モデルのCPU実装](https://github.com/TomoyukiSugiyama/transformer)からGPU移行するプロジェクトです。


## ドキュメント

詳細はフェーズ別・トピック別に `docs/` に整理してある:

| ドキュメント | 内容 |
|------------|------|
| [docs/roadmap.md](docs/roadmap.md) | 今後の改善案 / ロードマップ |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | ベンチマーク |
| [docs/g1-1.md](docs/g1-1.md) | プロジェクト骨格 |
| [docs/g1-2.md](docs/g1-2.md) | Matmul カーネル |
| [docs/g1-3.md](docs/g1-3.md) | Attention カーネル |
| [docs/g1-4.md](docs/g1-4.md) | Block 統合 |
| [docs/g1-5.md](docs/g1-5.md) | 学習ループ |


## ディレクトリ構成

```
src/
|-- gpu_context.rs
|-- kernel/     # Rust: wgpu のバッファ・パイプライン管理
|   |-- adam_w.rs
|   |-- attention.rs
|   |-- cross_entropy_loss.rs
|   |-- embedding.rs
|   |-- matmul.rs
|   |-- mod.rs
|   |-- residual_add.rs
|   |-- rms_norm.rs
|   |-- rope.rs
|   `-- swiglu_elementwise.rs
|-- lib.rs
|-- main.rs
|-- model/      # Transformer のブロック構成（kernel を組み合わせる）
|   |-- attention.rs
|   |-- ffn.rs
|   |-- language_model.rs
|   |-- mod.rs
|   `-- transformer_block.rs
|-- model_config.rs
|-- shader/     # WGSL: GPU 上で動く計算本体
|   |-- attention.wgsl
|   |-- cross_entropy_loss.wgsl
|   |-- embedding_backward.wgsl
|   |-- embedding.wgsl
|   |-- flash_attention.wgsl
|   |-- matmul.wgsl
|   |-- residual_add.wgsl
|   |-- rms_norm_backward.wgsl
|   |-- rms_norm.wgsl
|   |-- rope_backward.wgsl
|   |-- rope.wgsl
|   |-- swiglu_elementwise_backward.wgsl
|   `-- swiglu_elementwise.wgsl
|-- test_utils.rs
`-- util.rs
benches/        # ベンチマーク
|-- kernels.rs
`-- model.rs
```
