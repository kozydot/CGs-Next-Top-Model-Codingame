// https://www.codingame.com/ide/puzzle/cgs-next-top-model

use std::io;

macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}

// quick pair parser
fn parse_pairs(s: &str) -> Vec<(f64,f64)> {
    let toks: Vec<&str> = s.split_whitespace().collect();
    let mut v = Vec::with_capacity(toks.len()/2);
    let mut i = 0;
    while i+1 < toks.len() {
        let x = toks[i].parse::<f64>().unwrap_or(0.0);
        let y = toks[i+1].parse::<f64>().unwrap_or(0.0);
        v.push((x,y));
        i += 2;
    }
    v
}

// lightweight fast rng (xorshift64*)
struct FastRng { state: u64 }
impl FastRng {
    fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(0x9e3779b97f4a7c15) } }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa
        ((self.next_u64() >> 11) as f64) / ((1u64<<53) as f64)
    }
    #[inline]
    fn signed_unit(&mut self) -> f64 { 2.0 * self.next_f64() - 1.0 }
    #[inline]
    fn range_f64(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi-lo) * self.next_f64() }
}

const BIG: f64 = 1e300;

// precomputed dataset container to avoid indexing overhead
struct Data {
    xs: Vec<f64>,
    ys: Vec<f64>,
    x2: Vec<f64>,
}
impl Data {
    fn from_pairs(p: &[(f64,f64)]) -> Self {
        let n = p.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        for &(x,y) in p {
            xs.push(x);
            ys.push(y);
            x2.push(x*x);
        }
        Self { xs, ys, x2 }
    }
}

// sse evals with early cutoff (faster)
#[inline]
fn sse_linear_params(a: f64, b: f64, data: &Data, cutoff: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..data.xs.len() {
        let fx = a * data.xs[i] + b;
        let e = data.ys[i] - fx;
        s += e*e;
        if s > cutoff { return s; }
    }
    s
}
#[inline]
fn sse_parab_params(a: f64, b: f64, c: f64, data: &Data, cutoff: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..data.xs.len() {
        let fx = a * data.x2[i] + b * data.xs[i] + c;
        let e = data.ys[i] - fx;
        s += e*e;
        if s > cutoff { return s; }
    }
    s
}
#[inline]
fn sse_sine_params(a: f64, b: f64, c: f64, d: f64, data: &Data, cutoff: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..data.xs.len() {
        let fx = a * (b * data.xs[i] + c).sin() + d;
        let e = data.ys[i] - fx;
        s += e*e;
        if s > cutoff { return s; }
    }
    s
}
#[inline]
fn sse_exp_params(a: f64, b: f64, c: f64, data: &Data, cutoff: f64) -> f64 {
    let a = a.abs(); // spec says abs(a)^(x+b) + c
    let mut s = 0.0;
    for i in 0..data.xs.len() {
        let eexp = data.xs[i] + b;
        let fx = if a == 0.0 {
            // 0^t is 0 for t>0, 1 for t==0, invalid for t<0 -> treat as huge error
            if eexp > 0.0 { 0.0 + c } else if eexp == 0.0 { 1.0 + c } else { return BIG }
        } else {
            let v = a.powf(eexp);
            if !v.is_finite() || v.abs() > 1e200 { return BIG }
            v + c
        };
        let e = data.ys[i] - fx;
        s += e*e;
        if s > cutoff { return s; }
    }
    s
}

// helper to maintain tiny sorted top-k list (ascending by score)
fn push_topk(scores: &mut Vec<(f64, Vec<f64>)>, k: usize, entry: (f64, Vec<f64>)) {
    // cheap path: push then insertion sort back
    scores.push(entry);
    let mut j = scores.len()-1;
    while j > 0 && scores[j].0 < scores[j-1].0 {
        scores.swap(j, j-1);
        j -= 1;
    }
    if scores.len() > k { scores.truncate(k); }
}

// generic optimizer but tuned for speed
fn optimize_generic(
    rng: &mut FastRng,
    data: &Data,
    dims: usize,
    lo: &[f64],
    hi: &[f64],
    eval_kind: u8, // 0=line,1=parab,2=sine,3=exp
) -> Vec<f64> {
    // fast config: fewer samples but strong pruning + short local refine
    let base_samples = match dims {
        2 => 1200usize, // linear
        3 => 1600usize, // parab / exp
        4 => 2200usize, // sine
        _ => 1500usize
    };
    let top_k = 6usize;

    let mut best: Vec<(f64, Vec<f64>)> = Vec::new();
    // initial worst cutoff is INF, but we'll use last best as cutoff to prune
    let mut worst_cutoff = BIG;

    // reuse param vecs to avoid allocs inside loop as much as possible
    let mut params = vec![0.0f64; dims];

    for _ in 0..base_samples {
        for i in 0..dims {
            params[i] = rng.range_f64(lo[i], hi[i]);
        }
        let s = match eval_kind {
            0 => sse_linear_params(params[0], params[1], data, worst_cutoff),
            1 => sse_parab_params(params[0], params[1], params[2], data, worst_cutoff),
            2 => sse_sine_params(params[0], params[1], params[2], params[3], data, worst_cutoff),
            3 => sse_exp_params(params[0], params[1], params[2], data, worst_cutoff),
            _ => BIG
        };
        if !s.is_finite() { continue; }
        if best.len() < top_k {
            push_topk(&mut best, top_k, (s, params.clone()));
            if best.len() == top_k { worst_cutoff = best[top_k-1].0; }
        } else if s < best[top_k-1].0 {
            push_topk(&mut best, top_k, (s, params.clone()));
            worst_cutoff = best[top_k-1].0;
        }
    }

    // if nothing found, return mid points
    if best.is_empty() {
        let mut mid = vec![0.0f64; dims];
        for i in 0..dims { mid[i] = 0.5*(lo[i]+hi[i]); }
        return mid;
    }

    // local refine: few rounds, in-place tweaks, keep top_k seeds
    let mut global_best = best[0].1.clone();
    let mut global_score = best[0].0;

    // initial per-dim scales (half-range)
    let mut scales = vec![0.0; dims];
    for i in 0..dims { scales[i] = 0.5*(hi[i] - lo[i]); }

    for round in 0..5usize {
        // reduce factor quickly
        let factor = 0.5f64.powi(round as i32);
        let iters = 80usize / (1 + round); // fewer iters as rounds go
        let mut next_best: Vec<(f64, Vec<f64>)> = Vec::new();

        for &(s0, ref seed_p) in &best {
            // start from seed
            let mut cur_p = seed_p.clone();
            let mut cur_s = s0;
            // small hill-climb
            for _ in 0..iters {
                // perturb in-place into cand
                for i in 0..dims {
                    let delta = rng.signed_unit() * scales[i] * factor;
                    let mut v = cur_p[i] + delta;
                    if v < lo[i] { v = lo[i]; } else if v > hi[i] { v = hi[i]; }
                    params[i] = v;
                }
                let s = match eval_kind {
                    0 => sse_linear_params(params[0], params[1], data, cur_s),
                    1 => sse_parab_params(params[0], params[1], params[2], data, cur_s),
                    2 => sse_sine_params(params[0], params[1], params[2], params[3], data, cur_s),
                    3 => sse_exp_params(params[0], params[1], params[2], data, cur_s),
                    _ => BIG
                };
                if s.is_finite() && s < cur_s {
                    cur_s = s;
                    cur_p.copy_from_slice(&params);
                    if cur_s < global_score {
                        global_score = cur_s;
                        global_best = cur_p.clone();
                    }
                }
            }
            push_topk(&mut next_best, top_k, (cur_s, cur_p));
        }
        best = next_best;
    }

    global_best
}

fn main() {
    // read two lines
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let t = input_line.trim_matches('\n').to_string();
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let u = input_line.trim_matches('\n').to_string();

    let train_pairs = parse_pairs(&t);
    let test_pairs = parse_pairs(&u);

    let train = Data::from_pairs(&train_pairs);
    let test = Data::from_pairs(&test_pairs);

    // seed rng from data (deterministic)
    let mut seed: u64 = 1469598103934665603u64;
    for &(x,y) in &train_pairs {
        seed = seed.wrapping_mul(1099511628211u64).wrapping_add(x.to_bits());
        seed = seed.wrapping_mul(1099511628211u64).wrapping_add(y.to_bits());
    }
    for &(x,y) in &test_pairs {
        seed = seed.wrapping_mul(1099511628211u64).wrapping_add(x.to_bits());
        seed = seed.wrapping_mul(1099511628211u64).wrapping_add(y.to_bits());
    }
    let mut rng = FastRng::new(seed);

    // bounds for models
    let lin_lo = [-20.0, -20.0];
    let lin_hi = [20.0, 20.0];
    let par_lo = [-20.0, -20.0, -20.0];
    let par_hi = [20.0, 20.0, 20.0];
    let sin_lo = [-20.0, -20.0, -20.0, -20.0];
    let sin_hi = [20.0, 20.0, 20.0, 20.0];
    let exp_lo = [0.0, -20.0, -20.0];
    let exp_hi = [4.0, 20.0, 20.0];

    // optimize each model (fast)
    let best_lin = optimize_generic(&mut rng, &train, 2, &lin_lo, &lin_hi, 0);
    let best_par = optimize_generic(&mut rng, &train, 3, &par_lo, &par_hi, 1);
    let best_sin = optimize_generic(&mut rng, &train, 4, &sin_lo, &sin_hi, 2);
    let best_exp = optimize_generic(&mut rng, &train, 3, &exp_lo, &exp_hi, 3);

    // eval on test (no cutoff now)
    let s_lin = sse_linear_params(best_lin[0], best_lin[1], &test, BIG);
    let s_par = sse_parab_params(best_par[0], best_par[1], best_par[2], &test, BIG);
    let s_sin = sse_sine_params(best_sin[0], best_sin[1], best_sin[2], best_sin[3], &test, BIG);
    let s_exp = sse_exp_params(best_exp[0], best_exp[1], best_exp[2], &test, BIG);

    // pick winner
    let mut best_name = "LINEAR";
    let mut best_score = s_lin;
    if s_par < best_score { best_score = s_par; best_name = "PARABOLA"; }
    if s_sin < best_score { best_score = s_sin; best_name = "SINE"; }
    if s_exp < best_score { best_score = s_exp; best_name = "EXPONENTIAL"; }

    println!("{}", best_name);
}
