// Feed-forward portion of the published Hybrid family. The map encoder's
// cached embedding and the 27 derived features are inputs to this kernel.
use std::arch::wasm32::*;
pub struct Net {
    weights: Vec<f32>,
    offsets: [u32; 24],
    inbox: [f32; 1193],
    out: [f32; 18],
}
#[no_mangle]
pub extern "C" fn pn_new(size: usize) -> *mut Net {
    Box::into_raw(Box::new(Net {
        weights: vec![0.; size],
        offsets: [u32::MAX; 24],
        inbox: [0.; 1193],
        out: [0.; 18],
    }))
}
#[no_mangle]
pub unsafe extern "C" fn pn_weights(n: *mut Net) -> *mut f32 {
    (*n).weights.as_mut_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn pn_offsets(n: *mut Net) -> *mut u32 {
    (*n).offsets.as_mut_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn pn_input(n: *mut Net) -> *mut f32 {
    (*n).inbox.as_mut_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn pn_free(n: *mut Net) {
    if !n.is_null() {
        drop(Box::from_raw(n));
    }
}
#[inline(always)]
fn dot(x: &[f32], w: &[f32], b: f32) -> f32 {
    let mut k = 0;
    let mut acc = f32x4_splat(0.);
    let mut acc2 = f32x4_splat(0.);
    unsafe {
        while k + 8 <= x.len() {
            let a = v128_load(x.as_ptr().add(k) as *const v128);
            let u = v128_load(w.as_ptr().add(k) as *const v128);
            let c = v128_load(x.as_ptr().add(k + 4) as *const v128);
            let v = v128_load(w.as_ptr().add(k + 4) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(a, u));
            acc2 = f32x4_add(acc2, f32x4_mul(c, v));
            k += 8;
        }
    }
    acc = f32x4_add(acc, acc2);
    let mut sum = b
        + f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    while k < x.len() {
        sum += x[k] * w[k];
        k += 1;
    }
    sum
}
fn dense(input: &[f32], weights: &[f32], offs: &[u32], layer: usize, out: &mut [f32], act: u8) {
    let w = offs[layer] as usize;
    let b = offs[layer + 1] as usize;
    let n = input.len();
    for (i, v) in out.iter_mut().enumerate() {
        let sum = dot(input, &weights[w + i * n..w + (i + 1) * n], weights[b + i]);
        *v = match act {
            1 => sum.max(0.),
            2 => sum.tanh(),
            _ => sum,
        };
    }
}
#[no_mangle]
pub unsafe extern "C" fn pn_infer(np: *mut Net) -> *const f32 {
    let n = &mut *np;
    let v = &n.inbox;
    let w = &n.weights;
    let a = &n.offsets;
    let mut features = [0.; 320];
    features[..128].copy_from_slice(&v[1038..1166]);
    let mut scalars = [0.; 88];
    scalars[..60].copy_from_slice(&v[840..900]);
    scalars[60..].copy_from_slice(&v[1000..1028]);
    dense(&scalars, w, a, 4, &mut features[128..256], 2);
    let mut count = 0;
    for i in 0..10 {
        if v[1028 + i] < 0.5 {
            continue;
        }
        let mut b1 = [0.; 32];
        let mut b2 = [0.; 32];
        let k = 900 + 10 * i;
        dense(&v[k..k + 10], w, a, 0, &mut b1, 1);
        dense(&b1, w, a, 2, &mut b2, 1);
        for j in 0..32 {
            features[256 + j] += b2[j];
            features[288 + j] = features[288 + j].max(b2[j]);
        }
        count += 1;
    }
    if count > 0 {
        for v in &mut features[256..288] {
            *v /= count as f32;
        }
    }
    let mut hidden = [0.; 256];
    dense(&features, w, a, 6, &mut hidden, 2);
    if a[8] != u32::MAX {
        let mut delta = [0.; 256];
        dense(&v[1166..1193], w, a, 8, &mut delta, 0);
        for i in 0..256 {
            hidden[i] += delta[i];
        }
    }
    dense(&hidden, w, a, 10, &mut n.out, 0);
    let mut head = [0.; 64];
    let mut ds = [0.; 1];
    dense(&hidden, w, a, 12, &mut head, 2);
    dense(&head, w, a, 14, &mut ds, 0);
    let dodge = w[a[20] as usize] + ds[0];
    let mut av = [0.; 4];
    dense(&hidden, w, a, 16, &mut head, 2);
    dense(&head, w, a, 18, &mut av, 0);
    for i in 0..4 {
        av[i] += w[a[21] as usize + i];
    }
    let ammo = v[863];
    let soon = v[890] * (1. - (v[893] * 3.).clamp(0., 1.));
    let fire =
        av[0] * ammo + av[1] * soon - av[2] * (1. - ammo).powi(2) * (1. - soon) - av[3] * v[891];
    for i in 0..18 {
        n.out[i] += dodge * v[1018 + i / 2];
        if i % 2 == 1 {
            n.out[i] += fire;
        }
    }
    let idle = ((v[1027] * 25. - 8.) / 17.).clamp(0., 1.) * w[a[22] as usize];
    n.out[8] -= idle;
    n.out[9] -= idle;
    n.out.as_ptr()
}
