# 今後の改善案 / ロードマップ

## Phase G-1: M1 Max (Metal) で実装・検証
- wgpu で GPU カーネル (Vulkan/CUDA バックエンドに切替可能)
- Matmul / Attention の GPU 化

### G-1-1 プロジェクト骨格 ✅ 完了
Cargo.toml（wgpu + bytemuck + pollster）、ディレクトリ構成（src/kernel/, src/shader/, src/model/）、GPU デバイス初期化の動作確認。

-> [G-1-1](g1-1.md)

利用するクレート
| クレート     | 役割                                                                                  | いつ使う                 |
| -------- | ----------------------------------------------------------------------------------- | -------------------- |
| wgpu     | GPU API の抽象層。WGSL シェーダーをコンパイル・実行し、バッファ管理・コマンド発行を担う。Apple Silicon では Metal バックエンドで動作 | GPU カーネルの実装・実行全般     |
| bytemuck | Rust 構造体 → &[u8] 変換                                                                 | GPU バッファへのデータ転送時     |
| pollster | async fn を同期的にブロック実行                                                                | デバイス初期化・GPU 結果の読み戻し時 |

### G-1-2 Matmul カーネル ✅ 完了
WGSL で Tiled GEMM 実装、CPU 実装と数値一致確認、f32 / f16 両対応の検討。

-> [G-1-2](g1-2.md)

### G-1-3 Attention カーネル ✅ 完了
QK^T → scale → causal mask → softmax → V の一連。Fused Attention（FlashAttention）は G-1-3b として実施。

-> [G-1-3](g1-3.md)

### G-1-4 Block 統合 ✅ 完了
RMSNorm、SwiGLU FFN、Residual Add、RoPE を GPU 化。TransformerBlock 単体で CPU 版と出力一致を確認。

-> [G-1-4](g1-4.md)

### G-1-5 学習ループ
backward カーネル実装、AdamW の GPU 化、loss 計算。CPU 版より複雑なので最も工数がかかるフェーズ。

-> [G-1-5](g1-5.md)

### G-1-6 推論
Tokenizer、KV cache、GQA 実装。top-k、top-p を追加し推論

## Phase G-2: クラウド GPU サーバに移植
- NVIDIA A100/H100 (CUDA) に移行

```
G-2        クラウド GPU (CUDA) に移植
 ├─ G-2-1  CUDA バックエンド切替・動作確認
 └─ G-2-2  A100/H100 でのスループット計測
```