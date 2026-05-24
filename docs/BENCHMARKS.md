# G-1-4b ベンチマーク追加時の結果

- 初回: default 設定
- warm_up後 : `kernels.rs` に`warm_up_time(Duration::from_secs(3))` 追加
- measurement_time後 : `model.rs` に `warm_up_time(Duration::from_secs(3))` と `measurement_time(Duration::from_secs(10))` 追加

| カーネル                  | 初回 Median | warm_up後 Median | measurement_time後 Median |
| --------------------- | --------- | --------------- | ------------------------ |
| matmul_gpu            | 1.74 ms   | 1.88 ms         | 1.92 ms                  |
| attention_gpu         | 8.97 ms   | 8.48 ms         | 8.46 ms                  |
| transformer_block_gpu | 96.4 ms   | 85.2 ms         | 98.2 ms                  |

macOS の Metal バックエンドは長時間連続実行でサーマルスロットリングが起きやすく、これが原因で悪化と考えられる。

現状の判断としては warm_up_time=3s、measurement_time はデフォルト（5s）が最もノイズが少ない結果。

---

Transformer Block は内部で、
- attention を n_heads=4 回
- matmul（Q/K/V/O/gate/up/down の7回）
実行する。

```
attention × 4 heads  →  8.5ms × 4  =  34ms   (40%)
matmul × 7回         →  1.9ms × 7  =  13ms   (15%)
残り（overhead等）   →            ≈  38ms   (45%)
```

残り38msは RMS Norm、RoPE、SwiGLU の elementwise カーネル群と GPU の submit/readback オーバーヘッド


## ベースライン確定値（warm_up=3s、measurement_time=5s）

g1_4b にベースラインを保存
```
cargo bench --bench kernels -- --save-baseline g1_4b
cargo bench --bench model -- --save-baseline g1_4b
```

ベンチマーク結果

| カーネル                                                  | Median    | Std. Dev. |
| ----------------------------------------------------- | --------- | --------- |
| matmul_gpu 512×64×64                                  | 1.975 ms  | 204 µs    |
| attention / flash_attention seq=512, d_head=64        | 8.366 ms  | 1.380 ms  |
| attention / before_flash_attention seq=512, d_head=64 | 3.782 ms  | 200 µs    |
| transformer_block_gpu seq=512                         | 101.95 ms | 12.11 ms  |


機能導入後の比較は以下を実行

```
cargo bench --bench kernels -- --baseline g1_4b
cargo bench --bench model -- --baseline g1_4b
```

## Flash Attention が遅い理由

Flash Attention が速くなるのは メモリ帯域がボトルネック になる場合。
seq=512, d_head=64 のサイズでは HBM が飽和しないため、before_flash の単純な並列度の高さが勝る。

|          | before_flash           | flash_attention  |
| -------- | ---------------------- | ---------------- |
| HBM 書き込み | seq² = 1MB（score バッファ） | なし               |
| 本来の優位性   | seq が大きい（2048+）場合      | seq=512 では差が出にくい |

