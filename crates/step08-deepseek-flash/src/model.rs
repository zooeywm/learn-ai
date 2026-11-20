use std::{array, collections::VecDeque, ops::Range};

use mini_core::Rng;

use crate::tape::{Id, Tape, Vector};

pub const ENCODER_LAYERS: usize = 20;
const LAYERS: usize = 40;
const DIM: usize = 8;
const HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const QUERY_RANK: usize = 4;
const OUT_RANK: usize = 2;
const HC: usize = 4;
const HC_EPS: f32 = 1e-6;
const WINDOW: usize = 8;
const INDEX_TOP_K: usize = 4;
const CANDIDATE_BLOCK: usize = 2;
const CANDIDATE_TOP_K: usize = 3;
const EXPERTS: usize = 12;
const ACTIVE_EXPERTS: usize = 6;
const EXPERT_DIM: usize = 4;
const HASH_HEADS: usize = 8;
const HASH_KEYS: usize = 3 * HASH_HEADS;
const MIN_HASH_BUCKETS: usize = 23;
const HASH_WIDTH: usize = 2;
const KV_SOURCES: [usize; 4] = [2, 8, 14, 20];
pub const MAX_CONTEXT: usize = 128;

struct Parameters {
    values: Vec<f32>,
    rng: Rng,
}

impl Parameters {
    fn add(&mut self, values: impl IntoIterator<Item = f32>) -> Range<Id> {
        let start = self.values.len();
        self.values.extend(values);
        start..self.values.len()
    }

    fn fill(&mut self, count: usize, value: f32) -> Range<Id> {
        self.add(vec![value; count])
    }

    fn weight(&mut self, count: usize, std: f32) -> Range<Id> {
        let values: Vec<_> = (0..count).map(|_| self.rng.normal(std)).collect();
        self.add(values)
    }
}

struct Linear {
    weight: Range<Id>,
    input: usize,
}

impl Linear {
    fn new(p: &mut Parameters, input: usize, output: usize, gain: f32) -> Self {
        Self {
            weight: p.weight(input * output, gain / (input as f32).sqrt()),
            input,
        }
    }

    fn forward(&self, tape: &mut Tape, x: &[Id]) -> Vector {
        let weights: Vector = self.weight.clone().collect();
        weights
            .chunks_exact(self.input)
            .map(|row| tape.dot(x, row))
            .collect()
    }
}

fn rms_norm(tape: &mut Tape, x: &[Id], gamma: Range<Id>) -> Vector {
    let inv = tape.rms_scale(x);
    x.iter()
        .zip(gamma)
        .map(|(&x, g)| {
            let y = tape.mul(x, inv);
            tape.mul(y, g)
        })
        .collect()
}

fn rope(tape: &mut Tape, x: &mut [Id], position: usize, inverse: bool) {
    // A single rotary pair in the miniature profile. The other dimensions are unrotated.
    let angle = if inverse {
        -(position as f32)
    } else {
        position as f32
    };
    let (sin, cos) = angle.sin_cos();
    let end = x.len();
    let (a, b) = (x[end - 2], x[end - 1]);
    let b_cos = tape.scale(b, cos);
    let a_cos = tape.scale(a, cos);
    x[end - 2] = tape.affine(a_cos, b, -sin);
    x[end - 1] = tape.affine(b_cos, a, sin);
}

type Streams = [Vector; HC];

struct Mix {
    next_pre: Vector,
    post: Vector,
    residual: [Vector; HC],
}

struct HyperConnection {
    projection: Linear,
    base: Range<Id>,
    scale: Range<Id>,
}

impl HyperConnection {
    fn new(p: &mut Parameters) -> Self {
        let projection = Linear::new(p, HC * DIM, (2 + HC) * HC, 0.1);
        let base = p.add((0..(2 + HC) * HC).map(|i| match i {
            0..HC => -1.1,
            _ if i < 2 * HC => 0.0,
            _ if (i - 2 * HC) / HC == (i - 2 * HC) % HC => 4.0,
            _ => 0.0,
        }));
        Self {
            projection,
            base,
            scale: p.fill(3, 0.01),
        }
    }

    fn coefficients(&self, tape: &mut Tape, streams: &Streams) -> Mix {
        let flat: Vector = streams.iter().flatten().copied().collect();
        let inv = tape.rms_scale(&flat);
        let mixes: Vector = self
            .projection
            .forward(tape, &flat)
            .into_iter()
            .zip(self.base.clone())
            .enumerate()
            .map(|(i, (x, bias))| {
                let x = tape.mul(x, inv);
                let x = tape.mul(x, self.scale.start + (i / HC).min(2));
                tape.add(x, bias)
            })
            .collect();
        let next_pre = mixes[..HC]
            .iter()
            .map(|&x| {
                let y = tape.sigmoid(x);
                tape.add_scalar(y, HC_EPS)
            })
            .collect();
        let post = mixes[HC..2 * HC]
            .iter()
            .map(|&x| {
                let y = tape.sigmoid(x);
                tape.scale(y, 2.0)
            })
            .collect();
        let mut residual: [Vector; HC] = array::from_fn(|i| {
            tape.softmax(&mixes[2 * HC + i * HC..2 * HC + (i + 1) * HC])
                .into_iter()
                .map(|x| tape.add_scalar(x, HC_EPS))
                .collect()
        });
        for iteration in 0..20 {
            if iteration != 0 {
                for row in &mut residual {
                    let sum = tape.sum(row);
                    let sum = tape.add_scalar(sum, HC_EPS);
                    for x in row {
                        *x = tape.div(*x, sum);
                    }
                }
            }
            for column in 0..HC {
                let entries: [Id; HC] = array::from_fn(|row| residual[row][column]);
                let sum = tape.sum(&entries);
                let sum = tape.add_scalar(sum, HC_EPS);
                for row in &mut residual {
                    row[column] = tape.div(row[column], sum);
                }
            }
        }
        Mix {
            next_pre,
            post,
            residual,
        }
    }
}

fn collapse(tape: &mut Tape, streams: &Streams, mix: &[Id]) -> Vector {
    (0..DIM)
        .map(|d| {
            let column: [Id; HC] = array::from_fn(|i| streams[i][d]);
            tape.dot(&column, mix)
        })
        .collect()
}

fn expand(tape: &mut Tape, streams: &Streams, branch: &[Id], mix: &Mix) -> Streams {
    array::from_fn(|out| {
        let weights: [Id; HC] = array::from_fn(|i| mix.residual[i][out]);
        (0..DIM)
            .map(|d| {
                let column: [Id; HC] = array::from_fn(|i| streams[i][d]);
                let residual = tape.dot(&column, &weights);
                let update = tape.mul(branch[d], mix.post[out]);
                tape.add(residual, update)
            })
            .collect()
    })
}

fn bucket_sizes() -> impl Iterator<Item = usize> {
    (MIN_HASH_BUCKETS..).filter(|&n| (2..).take_while(|d| d * d <= n).all(|d| n % d != 0))
}

struct Engram {
    table: Range<Id>,
    buckets: [usize; HASH_KEYS],
    multipliers: [u64; 4],
    wkv: Linear,
    q_weight: Range<Id>,
    k_weight: Range<Id>,
}

impl Engram {
    fn new(p: &mut Parameters, layer: usize, vocab_size: usize) -> Self {
        let mut primes = bucket_sizes().skip(if layer == 1 { 0 } else { HASH_KEYS });
        let buckets: [usize; HASH_KEYS] = array::from_fn(|_| primes.next().expect("prime sizes"));
        let mut hash_rng = Rng::new((10007 * layer) as u64);
        let bound = (i64::MAX as u64 / vocab_size as u64) / 2;
        let multipliers = array::from_fn(|_| 2 * (hash_rng.next_u64() % bound) + 1);
        Self {
            table: p.weight(buckets.iter().sum::<usize>() * HASH_WIDTH, 0.1),
            buckets,
            multipliers,
            wkv: Linear::new(p, HASH_KEYS * HASH_WIDTH, DIM * (HC + 1), 0.5),
            q_weight: p.fill(HC * DIM, 1.0),
            k_weight: p.fill(HC * DIM, 1.0),
        }
    }

    fn forward(&self, tape: &mut Tape, streams: &mut Streams, ids: &[usize], pos: usize) {
        let mut lookup = Vec::with_capacity(HASH_KEYS * HASH_WIDTH);
        let mut hash = 0;
        let mut offset = self.table.start;
        for (lookback, &multiplier) in self.multipliers.iter().enumerate() {
            let id = pos.checked_sub(lookback).map_or(0, |i| ids[i]);
            hash ^= id as u64 * multiplier;
            if lookback == 0 {
                continue;
            }
            for &prime in &self.buckets[(lookback - 1) * HASH_HEADS..lookback * HASH_HEADS] {
                let base = offset + (hash % prime as u64) as usize * HASH_WIDTH;
                lookup.extend(base..base + HASH_WIDTH);
                offset += prime * HASH_WIDTH;
            }
        }
        let kv = self.wkv.forward(tape, &lookup);
        let value = &kv[HC * DIM..];
        for (i, stream) in streams.iter_mut().enumerate() {
            let key = &kv[i * DIM..(i + 1) * DIM];
            let inv_stream = tape.rms_scale(stream);
            let inv_key = tape.rms_scale(key);
            let mut weighted = Vec::with_capacity(DIM);
            for (d, &x) in stream.iter().enumerate() {
                let w = tape.mul(
                    self.q_weight.start + i * DIM + d,
                    self.k_weight.start + i * DIM + d,
                );
                weighted.push(tape.mul(x, w));
            }
            let dot = tape.dot(&weighted, key);
            let inv = tape.mul(inv_stream, inv_key);
            let dot = tape.mul(dot, inv);
            let dot = tape.scale(dot, (DIM as f32).sqrt().recip());
            let signed = tape.signed_sqrt(dot);
            let gate = tape.sigmoid(signed);
            for (x, &v) in stream.iter_mut().zip(value) {
                let update = tape.mul(gate, v);
                *x = tape.add(*x, update);
            }
        }
    }
}

struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Expert {
    fn new(p: &mut Parameters) -> Self {
        Self {
            gate: Linear::new(p, DIM, EXPERT_DIM, 1.0),
            up: Linear::new(p, DIM, EXPERT_DIM, 1.0),
            down: Linear::new(p, EXPERT_DIM, DIM, 0.16),
        }
    }

    fn forward(&self, tape: &mut Tape, x: &[Id]) -> Vector {
        let gates = self.gate.forward(tape, x);
        let up = self.up.forward(tape, x);
        let h: Vector = gates
            .into_iter()
            .zip(up)
            .map(|(g, u)| {
                let g = tape.clamp(g, f32::NEG_INFINITY, 10.0);
                let u = tape.clamp(u, -10.0, 10.0);
                let sigmoid = tape.sigmoid(g);
                let silu = tape.mul(g, sigmoid);
                tape.mul(silu, u)
            })
            .collect();
        self.down.forward(tape, &h)
    }
}

struct Moe {
    router: Linear,
    experts: [Expert; EXPERTS],
    shared: Expert,
}

impl Moe {
    fn new(p: &mut Parameters) -> Self {
        Self {
            router: Linear::new(p, DIM, EXPERTS, 1.0),
            experts: array::from_fn(|_| Expert::new(p)),
            shared: Expert::new(p),
        }
    }

    fn forward(&self, tape: &mut Tape, x: &[Id]) -> Vector {
        let scores: Vector = self
            .router
            .forward(tape, x)
            .into_iter()
            .map(|s| tape.sqrt_softplus(s))
            .collect();
        let selected = top_k(
            (0..EXPERTS).map(|i| (i, tape.value(scores[i]))).collect(),
            ACTIVE_EXPERTS,
        );
        let chosen: Vector = selected.iter().map(|&i| scores[i]).collect();
        let total = tape.sum(&chosen);
        let mut output = self.shared.forward(tape, x);
        for id in selected {
            let normalized = tape.div(scores[id], total);
            let weight = tape.scale(normalized, 1.5);
            let values = self.experts[id].forward(tape, x);
            for (out, value) in output.iter_mut().zip(values) {
                let update = tape.mul(value, weight);
                *out = tape.add(*out, update);
            }
        }
        output
    }
}

fn top_k(mut scores: Vec<(usize, f32)>, k: usize) -> Vec<usize> {
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scores.truncate(k);
    scores.into_iter().map(|(i, _)| i).collect()
}

fn kv_source(layer: usize) -> Option<usize> {
    KV_SOURCES.iter().rposition(|&owner| layer >= owner)
}

struct Compressed {
    kv: Vector,
    key: Vec<f32>,
}

#[derive(Default)]
struct SharedCache {
    pending: Option<(Vector, Vector)>,
    entries: Vec<Compressed>,
}

struct Compressor {
    kv: Linear,
    pool: Option<Linear>,
    gamma: Range<Id>,
    index_k: Linear,
    index_gamma: Range<Id>,
}

impl Compressor {
    fn new(p: &mut Parameters, ratio: usize) -> Self {
        Self {
            kv: Linear::new(p, DIM, HEAD_DIM, 1.0),
            pool: (ratio == 2).then(|| Linear::new(p, DIM, HEAD_DIM, 1.0)),
            gamma: p.fill(HEAD_DIM, 1.0),
            index_k: Linear::new(p, HEAD_DIM, HEAD_DIM, 1.0),
            index_gamma: p.fill(HEAD_DIM, 1.0),
        }
    }

    fn append(&self, tape: &mut Tape, cache: &mut SharedCache, x: &[Id], pos: usize) {
        let mut kv = self.kv.forward(tape, x);
        let mut position = pos;
        if let Some(pool) = &self.pool {
            let scores = pool.forward(tape, x);
            let Some((previous, previous_scores)) = cache.pending.take() else {
                cache.pending = Some((kv, scores));
                return;
            };
            for d in 0..HEAD_DIM {
                let mix = tape.softmax(&[previous_scores[d], scores[d]]);
                kv[d] = tape.dot(&[previous[d], kv[d]], &mix);
            }
            position -= 1;
        }
        let mut kv = rms_norm(tape, &kv, self.gamma.clone());
        let key = self.index_k.forward(tape, &kv);
        let mut key = rms_norm(tape, &key, self.index_gamma.clone());
        rope(tape, &mut key, position, false);
        rope(tape, &mut kv, position, false);
        cache.entries.push(Compressed {
            kv,
            key: key.into_iter().map(|k| tape.value(k)).collect(),
        });
    }
}

#[derive(Clone, Copy)]
enum IndexStage {
    Encoder,
    CandidateSource,
    Reindex,
}

struct Indexer {
    q: Linear,
    weights: Linear,
    stage: IndexStage,
}

impl Indexer {
    fn new(p: &mut Parameters, stage: IndexStage) -> Self {
        Self {
            q: Linear::new(p, QUERY_RANK, HEADS * HEAD_DIM, 1.0),
            weights: Linear::new(p, DIM, HEADS, 1.0),
            stage,
        }
    }

    fn select(
        &self,
        tape: &mut Tape,
        x: &[Id],
        qr: &[Id],
        entries: &[Compressed],
        pos: usize,
        selection: &mut Selection,
    ) {
        let mut queries = self.q.forward(tape, qr);
        for head in queries.chunks_exact_mut(HEAD_DIM) {
            rope(tape, head, pos, false);
        }
        let weights = self.weights.forward(tape, x);
        // Hard ranking is discrete. The main language-model loss does not supervise index scores.
        let score = |entry: &Compressed| {
            queries
                .chunks_exact(HEAD_DIM)
                .zip(&weights)
                .map(|(head, &w)| {
                    let dot = head
                        .iter()
                        .zip(&entry.key)
                        .map(|(&q, &k)| tape.value(q) * k)
                        .sum::<f32>();
                    dot.max(0.0) * tape.value(w)
                })
                .sum::<f32>()
        };
        let scores: Vec<_> = entries.iter().map(score).enumerate().collect();
        if matches!(self.stage, IndexStage::CandidateSource) {
            let last = scores.len().saturating_sub(1) / CANDIDATE_BLOCK;
            let blocks: Vec<_> = scores
                .chunks(CANDIDATE_BLOCK)
                .enumerate()
                .map(|(i, block)| {
                    let max = block
                        .iter()
                        .map(|&(_, s)| s)
                        .fold(f32::NEG_INFINITY, f32::max);
                    (i, if i == last { f32::INFINITY } else { max })
                })
                .collect();
            selection.candidates = top_k(blocks, CANDIDATE_TOP_K);
        }
        let use_candidates = matches!(self.stage, IndexStage::Reindex);
        let scores = scores
            .into_iter()
            .filter(|&(i, _)| {
                !use_candidates || selection.candidates.contains(&(i / CANDIDATE_BLOCK))
            })
            .collect();
        selection.positions = top_k(scores, INDEX_TOP_K);
    }
}

#[derive(Default)]
struct Selection {
    candidates: Vec<usize>,
    positions: Vec<usize>,
}

struct Attention {
    q_a: Linear,
    q_gamma: Range<Id>,
    q_b: Linear,
    local_kv: Linear,
    kv_gamma: Range<Id>,
    sink: Range<Id>,
    output_groups: [Linear; HEADS],
    output: Linear,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
}

impl Attention {
    fn new(p: &mut Parameters, layer: usize) -> Self {
        let compressor = KV_SOURCES
            .contains(&layer)
            .then(|| Compressor::new(p, if layer < ENCODER_LAYERS { 2 } else { 1 }));
        let indexer = match layer {
            2 | 8 | 14 => Some(Indexer::new(p, IndexStage::Encoder)),
            ENCODER_LAYERS => Some(Indexer::new(p, IndexStage::CandidateSource)),
            24 | 28 | 32 | 36 => Some(Indexer::new(p, IndexStage::Reindex)),
            _ => None,
        };
        Self {
            q_a: Linear::new(p, DIM, QUERY_RANK, 1.0),
            q_gamma: p.fill(QUERY_RANK, 1.0),
            q_b: Linear::new(p, QUERY_RANK, HEADS * HEAD_DIM, 1.0),
            local_kv: Linear::new(p, DIM, HEAD_DIM, 1.0),
            kv_gamma: p.fill(HEAD_DIM, 1.0),
            sink: p.fill(HEADS, 0.0),
            output_groups: array::from_fn(|_| Linear::new(p, HEAD_DIM, OUT_RANK, 1.0)),
            output: Linear::new(p, HEADS * OUT_RANK, DIM, 0.15),
            compressor,
            indexer,
        }
    }

    fn forward(
        &self,
        tape: &mut Tape,
        x: &[Id],
        history: &mut History,
        selection: &mut Selection,
        layer: usize,
        pos: usize,
    ) -> Vector {
        let qr = self.q_a.forward(tape, x);
        let qr = rms_norm(tape, &qr, self.q_gamma.clone());
        let queries = self.q_b.forward(tape, &qr);
        let local = self.local_kv.forward(tape, x);
        let mut local = rms_norm(tape, &local, self.kv_gamma.clone());
        rope(tape, &mut local, pos, false);
        let window = &mut history.windows[layer];
        window.push_back(local);
        if window.len() > WINDOW {
            window.pop_front();
        }

        let mut memory: Vec<_> = window.iter().map(Vector::as_slice).collect();
        if let Some(source) = kv_source(layer) {
            let cache = &mut history.global[source];
            if let Some(compressor) = &self.compressor {
                // Decoder-global is prepared once at the encoder/decoder boundary.
                if layer != ENCODER_LAYERS {
                    compressor.append(tape, cache, x, pos);
                }
            }
            let visible = if layer < ENCODER_LAYERS {
                pos.div_ceil(2)
            } else {
                pos + 1
            };
            let entries = &cache.entries[..visible];
            if let Some(indexer) = &self.indexer {
                indexer.select(tape, x, &qr, entries, pos, selection);
            }
            memory.extend(
                selection
                    .positions
                    .iter()
                    .map(|&i| entries[i].kv.as_slice()),
            );
        }

        let mut groups = Vec::with_capacity(HEADS * OUT_RANK);
        for (head, q) in queries.chunks_exact(HEAD_DIM).enumerate() {
            let mut q = q.to_vec();
            rope(tape, &mut q, pos, false);
            let mut logits: Vector = memory
                .iter()
                .map(|k| {
                    let dot = tape.dot(&q, k);
                    tape.scale(dot, (HEAD_DIM as f32).sqrt().recip())
                })
                .collect();
            logits.push(self.sink.start + head);
            let weights = tape.softmax(&logits);
            let mut output: Vector = (0..HEAD_DIM)
                .map(|d| {
                    let column: Vector = memory.iter().map(|kv| kv[d]).collect();
                    tape.dot(&column, &weights[..memory.len()])
                })
                .collect();
            rope(tape, &mut output, pos, true);
            groups.extend(self.output_groups[head].forward(tape, &output));
        }
        self.output.forward(tape, &groups)
    }
}

struct Layer {
    id: usize,
    attn_hc: HyperConnection,
    moe_hc: HyperConnection,
    attn_gamma: Range<Id>,
    moe_gamma: Range<Id>,
    attn: Attention,
    moe: Moe,
    engram: Option<Engram>,
}

impl Layer {
    fn new(p: &mut Parameters, id: usize, vocab_size: usize) -> Self {
        Self {
            id,
            attn_hc: HyperConnection::new(p),
            moe_hc: HyperConnection::new(p),
            attn_gamma: p.fill(DIM, 1.0),
            moe_gamma: p.fill(DIM, 1.0),
            attn: Attention::new(p, id),
            moe: Moe::new(p),
            engram: [1, 14]
                .contains(&id)
                .then(|| Engram::new(p, id, vocab_size)),
        }
    }

    fn attn_input(&self, tape: &mut Tape, frame: &Frame) -> Vector {
        let x = collapse(tape, &frame.streams, &frame.pre_mix);
        rms_norm(tape, &x, self.attn_gamma.clone())
    }

    fn forward(
        &self,
        tape: &mut Tape,
        mut frame: Frame,
        history: &mut History,
        selection: &mut Selection,
        ids: &[usize],
        pos: usize,
    ) -> Frame {
        if let Some(engram) = &self.engram {
            engram.forward(tape, &mut frame.streams, ids, pos);
        }
        let attn_mix = self.attn_hc.coefficients(tape, &frame.streams);
        let x = self.attn_input(tape, &frame);
        let update = self
            .attn
            .forward(tape, &x, history, selection, self.id, pos);
        let streams = expand(tape, &frame.streams, &update, &attn_mix);
        let moe_mix = self.moe_hc.coefficients(tape, &streams);
        let x = collapse(tape, &streams, &attn_mix.next_pre);
        let x = rms_norm(tape, &x, self.moe_gamma.clone());
        let update = self.moe.forward(tape, &x);
        Frame {
            streams: expand(tape, &streams, &update, &moe_mix),
            pre_mix: moe_mix.next_pre,
        }
    }
}

struct Frame {
    streams: Streams,
    pre_mix: Vector,
}

pub struct History {
    windows: [VecDeque<Vector>; LAYERS],
    global: [SharedCache; 4],
}

impl History {
    pub fn new() -> Self {
        Self {
            windows: array::from_fn(|_| VecDeque::new()),
            global: array::from_fn(|_| SharedCache::default()),
        }
    }

    pub fn cache_positions(&self) -> (usize, usize) {
        (
            self.windows.iter().map(VecDeque::len).sum(),
            self.global.iter().map(|c| c.entries.len()).sum(),
        )
    }
}

pub struct Model {
    pub weights: Vec<f32>,
    vocab: Range<Id>,
    layers: Vec<Layer>,
    norm: Range<Id>,
    head: Linear,
}

impl Model {
    pub fn new(vocab_size: usize) -> Self {
        let mut p = Parameters {
            values: Vec::new(),
            rng: Rng::new(42),
        };
        let vocab = p.weight(vocab_size * DIM, 0.3);
        let layers = (0..LAYERS)
            .map(|id| Layer::new(&mut p, id, vocab_size))
            .collect();
        let norm = p.fill(DIM, 1.0);
        let head = Linear::new(&mut p, DIM, vocab_size, 0.7);
        Self {
            weights: p.values,
            vocab,
            layers,
            norm,
            head,
        }
    }

    fn encode(&self, tape: &mut Tape, history: &mut History, ids: &[usize], pos: usize) -> Frame {
        let offset = self.vocab.start + ids[pos] * DIM;
        let hidden: Vector = (offset..offset + DIM).collect();
        let mut frame = Frame {
            streams: array::from_fn(|_| hidden.clone()),
            pre_mix: (0..HC).map(|i| tape.constant(f32::from(i == 0))).collect(),
        };
        let mut selection = Selection::default();
        for layer in &self.layers[..ENCODER_LAYERS] {
            frame = layer.forward(tape, frame, history, &mut selection, ids, pos);
        }
        let decoder_entry = &self.layers[ENCODER_LAYERS];
        let x = decoder_entry.attn_input(tape, &frame);
        let compressor = decoder_entry
            .attn
            .compressor
            .as_ref()
            .expect("decoder KV owner");
        compressor.append(tape, &mut history.global[3], &x, pos);
        frame
    }

    fn decode(
        &self,
        tape: &mut Tape,
        history: &mut History,
        ids: &[usize],
        mut frame: Frame,
        pos: usize,
    ) -> Frame {
        let mut selection = Selection::default();
        for layer in &self.layers[ENCODER_LAYERS..] {
            frame = layer.forward(tape, frame, history, &mut selection, ids, pos);
        }
        frame
    }

    fn logits(&self, tape: &mut Tape, frame: &Frame) -> Vector {
        let x = collapse(tape, &frame.streams, &frame.pre_mix);
        let x = rms_norm(tape, &x, self.norm.clone());
        self.head.forward(tape, &x)
    }

    pub fn prefill(&self, tape: &mut Tape, history: &mut History, ids: &[usize]) -> Vector {
        let encoded: Vec<_> = (0..ids.len())
            .map(|pos| self.encode(tape, history, ids, pos))
            .collect();
        // A cold prefill warms every decoder window. Later decode steps only append one token.
        let mut last = None;
        for (pos, frame) in encoded.into_iter().enumerate() {
            last = Some(self.decode(tape, history, ids, frame, pos));
        }
        self.logits(tape, &last.expect("prefill requires at least one token"))
    }

    pub fn decode_step(&self, tape: &mut Tape, history: &mut History, ids: &[usize]) -> Vector {
        let pos = ids.len() - 1;
        let encoded = self.encode(tape, history, ids, pos);
        let decoded = self.decode(tape, history, ids, encoded, pos);
        self.logits(tape, &decoded)
    }

    pub fn loss(&self, tape: &mut Tape, ids: &[usize], answer_start: usize) -> Id {
        let mut history = History::new();
        let mut losses = Vec::new();
        for pos in 0..ids.len() - 1 {
            let frame = self.encode(tape, &mut history, ids, pos);
            let frame = self.decode(tape, &mut history, ids, frame, pos);
            if pos + 1 >= answer_start {
                let logits = self.logits(tape, &frame);
                losses.push(tape.cross_entropy(&logits, ids[pos + 1]));
            }
        }
        let total = tape.sum(&losses);
        tape.scale(total, 1.0 / losses.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    fn logits(tape: &Tape, ids: &[Id]) -> Vec<f32> {
        ids.iter().map(|&id| tape.value(id)).collect()
    }

    #[test]
    fn split_prefill_and_single_token_decode_agree_past_the_sliding_window() {
        let model = Model::new(32);
        let tokens: Vec<_> = (0..17).map(|i| (i * 7) % 32).collect();
        let mut tape = Tape::new(&model.weights, false);
        let mut history = History::new();
        model.prefill(&mut tape, &mut history, &tokens[..9]);
        let cached = (10..=tokens.len())
            .map(|len| model.decode_step(&mut tape, &mut history, &tokens[..len]))
            .last()
            .unwrap();
        assert_eq!(history.cache_positions(), (LAYERS * WINDOW, 3 * 8 + 17));

        let mut full_tape = Tape::new(&model.weights, false);
        let full = model.prefill(&mut full_tape, &mut History::new(), &tokens);
        compare(&logits(&tape, &cached), &logits(&full_tape, &full));
    }

    #[test]
    fn training_backpropagates_through_shared_kv_into_the_encoder() {
        let model = Model::new(32);
        let mut tape = Tape::new(&model.weights, true);
        let loss = model.loss(&mut tape, &[1, 12, 2, 17, 3], 3);
        let grad = tape.backward(loss, model.weights.len());
        assert!(grad.iter().all(|x| x.is_finite()));
        for range in [
            model.layers[19].attn.output.weight.clone(),
            model.layers[20]
                .attn
                .compressor
                .as_ref()
                .unwrap()
                .kv
                .weight
                .clone(),
            model.layers[39].attn.output.weight.clone(),
        ] {
            assert!(grad[range].iter().any(|g| g.abs() > 1e-8));
        }
    }

    #[test]
    fn future_globals_do_not_change_logits_of_an_earlier_frame() {
        let model = Model::new(32);
        let ids = [1, 5, 18, 11, 9, 4, 7];
        let mut tape = Tape::new(&model.weights, false);
        let mut history = History::new();
        let frames: Vec<_> = (0..ids.len())
            .map(|pos| model.encode(&mut tape, &mut history, &ids, pos))
            .collect();
        let mut earlier = Vec::new();
        for (pos, frame) in frames.into_iter().take(4).enumerate() {
            let frame = model.decode(&mut tape, &mut history, &ids, frame, pos);
            earlier = model.logits(&mut tape, &frame);
        }
        let mut short_tape = Tape::new(&model.weights, false);
        let short = model.prefill(&mut short_tape, &mut History::new(), &ids[..4]);
        compare(&logits(&tape, &earlier), &logits(&short_tape, &short));
    }
}
