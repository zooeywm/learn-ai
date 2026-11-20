//! Scalar reverse-mode autodiff for the small CPU reference model.
//! Each edge stores the local derivative used during the backward pass.

pub type Id = usize;
pub type Vector = Vec<Id>;

pub struct Tape {
    values: Vec<f32>,
    starts: Vec<usize>,
    edges: Vec<(Id, f32)>,
    training: bool,
}

impl Tape {
    pub fn new(parameters: &[f32], training: bool) -> Self {
        Self {
            values: parameters.to_vec(),
            starts: if training {
                vec![0; parameters.len()]
            } else {
                Vec::new()
            },
            edges: Vec::new(),
            training,
        }
    }

    pub fn value(&self, id: Id) -> f32 {
        self.values[id]
    }

    fn push(&mut self, value: f32, edges: impl IntoIterator<Item = (Id, f32)>) -> Id {
        let id = self.values.len();
        self.values.push(value);
        if self.training {
            self.starts.push(self.edges.len());
            self.edges.extend(edges);
        }
        id
    }

    pub fn constant(&mut self, value: f32) -> Id {
        self.push(value, [])
    }

    pub fn add(&mut self, a: Id, b: Id) -> Id {
        self.push(self.value(a) + self.value(b), [(a, 1.0), (b, 1.0)])
    }

    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        let (x, y) = (self.value(a), self.value(b));
        self.push(x * y, [(a, y), (b, x)])
    }

    pub fn scale(&mut self, x: Id, factor: f32) -> Id {
        self.push(self.value(x) * factor, [(x, factor)])
    }

    pub fn affine(&mut self, a: Id, b: Id, factor: f32) -> Id {
        self.push(
            self.value(a) + factor * self.value(b),
            [(a, 1.0), (b, factor)],
        )
    }

    pub fn add_scalar(&mut self, x: Id, offset: f32) -> Id {
        self.push(self.value(x) + offset, [(x, 1.0)])
    }

    pub fn div(&mut self, a: Id, b: Id) -> Id {
        let inv = self.value(b).recip();
        let value = self.value(a) * inv;
        self.push(value, [(a, inv), (b, -value * inv)])
    }

    pub fn exp(&mut self, x: Id) -> Id {
        let value = self.value(x).exp();
        self.push(value, [(x, value)])
    }

    pub fn ln(&mut self, x: Id) -> Id {
        self.push(self.value(x).ln(), [(x, self.value(x).recip())])
    }

    pub fn sigmoid(&mut self, x: Id) -> Id {
        let y = 1.0 / (1.0 + (-self.value(x)).exp());
        self.push(y, [(x, y * (1.0 - y))])
    }

    pub fn sqrt_softplus(&mut self, x: Id) -> Id {
        let x_val = self.value(x);
        let y = (x_val.max(0.0) + (-x_val.abs()).exp().ln_1p()).sqrt();
        self.push(y, [(x, 0.5 / y.max(1e-12) / (1.0 + (-x_val).exp()))])
    }

    pub fn signed_sqrt(&mut self, x: Id) -> Id {
        let x_val = self.value(x);
        let magnitude = x_val.abs().max(1e-6).sqrt();
        let grad = if x_val.abs() >= 1e-6 {
            0.5 / magnitude
        } else {
            0.0
        };
        self.push(magnitude.copysign(x_val), [(x, grad)])
    }

    pub fn clamp(&mut self, x: Id, min: f32, max: f32) -> Id {
        let v = self.value(x);
        self.push(
            v.clamp(min, max),
            [(x, f32::from((min..=max).contains(&v)))],
        )
    }

    pub fn sum(&mut self, xs: &[Id]) -> Id {
        let value = xs.iter().map(|&x| self.value(x)).sum();
        self.push(value, xs.iter().map(|&x| (x, 1.0)))
    }

    pub fn dot(&mut self, a: &[Id], b: &[Id]) -> Id {
        debug_assert_eq!(a.len(), b.len());
        let value = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| self.value(x) * self.value(y))
            .sum();
        let start = self.edges.len();
        if self.training {
            for (&x, &y) in a.iter().zip(b) {
                self.edges.push((x, self.value(y)));
                self.edges.push((y, self.value(x)));
            }
            self.starts.push(start);
        }
        self.values.push(value);
        self.values.len() - 1
    }

    pub fn rms_scale(&mut self, xs: &[Id]) -> Id {
        let n = xs.len() as f32;
        let ms = xs.iter().map(|&x| self.value(x).powi(2)).sum::<f32>() / n;
        let value = (ms + 1e-20).sqrt().recip();
        let start = self.edges.len();
        if self.training {
            let factor = -value.powi(3) / n;
            for &x in xs {
                self.edges.push((x, self.value(x) * factor));
            }
            self.starts.push(start);
        }
        self.values.push(value);
        self.values.len() - 1
    }

    pub fn softmax(&mut self, xs: &[Id]) -> Vector {
        let max = xs
            .iter()
            .map(|&x| self.value(x))
            .fold(f32::NEG_INFINITY, f32::max);
        let exp: Vector = xs
            .iter()
            .map(|&x| {
                let shifted = self.add_scalar(x, -max);
                self.exp(shifted)
            })
            .collect();
        let sum = self.sum(&exp);
        exp.into_iter().map(|x| self.div(x, sum)).collect()
    }

    pub fn cross_entropy(&mut self, logits: &[Id], target: usize) -> Id {
        let max = logits
            .iter()
            .map(|&x| self.value(x))
            .fold(f32::NEG_INFINITY, f32::max);
        let exp: Vector = logits
            .iter()
            .map(|&x| {
                let shifted = self.add_scalar(x, -max);
                self.exp(shifted)
            })
            .collect();
        let sum = self.sum(&exp);
        let log_sum = self.ln(sum);
        let shifted = self.add_scalar(log_sum, max);
        self.affine(shifted, logits[target], -1.0)
    }

    pub fn backward(&self, loss: Id, parameters: usize) -> Vec<f32> {
        assert!(self.training);
        let mut grad = vec![0.0; self.values.len()];
        grad[loss] = 1.0;
        for node in (parameters..=loss).rev() {
            let upstream = grad[node];
            if upstream != 0.0 {
                let end = self
                    .starts
                    .get(node + 1)
                    .copied()
                    .unwrap_or(self.edges.len());
                for &(input, local) in &self.edges[self.starts[node]..end] {
                    grad[input] += local * upstream;
                }
            }
        }
        grad.truncate(parameters);
        grad
    }
}
