use std::collections::HashMap;

use crate::checkpoint::{Checkpointable, WeightMap};

pub struct AdamWParam {
    data: Vec<f32>, // パラメータ本体(W1, W2, b など)
    m: Vec<f32>,    // 一次モーメント
    v: Vec<f32>,    // 二次モーメント
}

impl AdamWParam {
    pub fn new(data: Vec<f32>) -> Self {
        let n = data.len();
        Self {
            data,
            m: vec![0.0f32; n],
            v: vec![0.0f32; n],
        }
    }

    pub fn step(
        &mut self,
        grad: &[f32],
        t: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) {
        let t = t as f32;

        for i in 0..self.data.len() {
            let g = grad[i];
            // モーメント更新
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / (1.0 - beta1.powf(t));
            let v_hat = self.v[i] / (1.0 - beta2.powf(t));

            self.data[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * self.data[i]);
        }
    }
}

pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step_count: usize,
    params: HashMap<String, AdamWParam>,
    grad_scale: f32,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self::new_with_wd(lr, 0.01)
    }

    pub fn new_with_wd(lr: f32, wd: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd,
            step_count: 0,
            params: HashMap::new(),
            grad_scale: 1.0,
        }
    }

    pub fn set_wd(&mut self, wd: f32) {
        self.wd = wd;
    }

    pub fn set_beta2(&mut self, beta2: f32) {
        self.beta2 = beta2;
    }

    pub fn set_grad_scale(&mut self, batch_size: usize) {
        self.grad_scale = 1.0 / batch_size as f32;
    }

    pub fn reset_grad_scale(&mut self) {
        self.grad_scale = 1.0;
    }

    pub fn increment_step(&mut self) {
        self.step_count += 1;
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step(&mut self, param_id: &str, param: &mut Vec<f32>, grad: &[f32]) {
        assert_eq!(
            param.len(),
            grad.len(),
            "param/grad size mismatch: {param_id}"
        );

        let grad_norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        eprintln!(
            "  adam::step {param_id} grad_norm={grad_norm:.6} step={}",
            self.step_count
        );

        let scale = self.grad_scale;
        let scaled_grad: Vec<f32> = grad.iter().map(|&g| g * scale).collect();

        let entry = self
            .params
            .entry(param_id.to_string())
            .or_insert_with(|| AdamWParam::new(param.clone()));

        entry.step(
            &scaled_grad,
            self.step_count,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.wd,
        );

        param.copy_from_slice(&entry.data);
    }
}

impl Checkpointable for AdamW {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("lr", self.lr.to_bits() as u64);
        map.insert_scalar("beta1", self.beta1.to_bits() as u64);
        map.insert_scalar("beta2", self.beta2.to_bits() as u64);
        map.insert_scalar("eps", self.eps.to_bits() as u64);
        map.insert_scalar("wd", self.wd.to_bits() as u64);
        map.insert_scalar("step_count", self.step_count as u64);
        map.insert_scalar("grad_scale", self.grad_scale.to_bits() as u64);
        for (k, param) in &self.params {
            map.insert_vector(&format!("m.{k}"), param.m.clone());
            map.insert_vector(&format!("v.{k}"), param.v.clone());
            map.insert_vector(&format!("data.{k}"), param.data.clone());
        }
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> std::io::Result<()> {
        self.lr = f32::from_bits(map.get_scalar("lr")? as u32);
        self.beta1 = f32::from_bits(map.get_scalar("beta1")? as u32);
        self.beta2 = f32::from_bits(map.get_scalar("beta2")? as u32);
        self.eps = f32::from_bits(map.get_scalar("eps")? as u32);
        self.wd = f32::from_bits(map.get_scalar("wd")? as u32);
        self.step_count = map.get_scalar("step_count")? as usize;
        self.grad_scale = f32::from_bits(map.get_scalar("grad_scale")? as u32);
        for key in map.vector_keys() {
            if let Some(k) = key.strip_prefix("data.") {
                let data = map.get_vector(key)?.clone();
                let n = data.len();
                let entry = self
                    .params
                    .entry(k.to_string())
                    .or_insert_with(|| AdamWParam {
                        data: vec![0.0; n],
                        m: vec![0.0; n],
                        v: vec![0.0; n],
                    });
                entry.data = data;
            } else if let Some(k) = key.strip_prefix("m.") {
                let entry = self
                    .params
                    .entry(k.to_string())
                    .or_insert_with(|| AdamWParam {
                        data: vec![],
                        m: vec![],
                        v: vec![],
                    });
                entry.m = map.get_vector(key)?.clone();
            } else if let Some(k) = key.strip_prefix("v.") {
                let entry = self
                    .params
                    .entry(k.to_string())
                    .or_insert_with(|| AdamWParam {
                        data: vec![],
                        m: vec![],
                        v: vec![],
                    });
                entry.v = map.get_vector(key)?.clone();
            }
        }

        for (k, p) in &self.params {
            if p.data.len() != p.m.len() || p.data.len() != p.v.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("AdamW state size mismatch for {k}"),
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_step() {
        let mut opt = AdamW::new(0.001);
        let mut param = vec![1.0f32, 2.0f32];
        let grad = vec![0.1f32, 0.2f32];

        opt.increment_step();
        opt.step("w", &mut param, &grad);

        // step 後にパラメータが減少していることを確認
        assert!(param[0] < 1.0, "param[0] should decrease");
        assert!(param[1] < 2.0, "param[1] should decrease");
    }

    #[test]
    fn test_adamw_grad_scale() {
        let mut opt1 = AdamW::new(0.001);
        let mut opt2 = AdamW::new(0.001);

        // バッチサイズ 4 で平均
        opt2.set_grad_scale(4);

        let mut p1 = vec![1.0f32];
        let mut p2 = vec![1.0f32];
        let grad_large = vec![4.0f32];
        // 4.0 / 4 = 1.0 と等価
        let grad_small = vec![1.0f32];

        opt1.increment_step();
        opt2.increment_step();
        opt1.step("w", &mut p1, &grad_large);
        opt2.step("w", &mut p2, &grad_small);

        // grad_scale=1/4 で 4.0 を渡すのは、1.0 をそのまま渡すのと等価
        assert!((p1[0] - p2[0]).abs() < 1e-6);
    }
}
