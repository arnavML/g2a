//! A fuzz harness for the runtime's parsers and the shader translator.
//! See fuzz/README.md.
use std::panic::{catch_unwind, AssertUnwindSafe};
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: u64) -> u64 { self.next() % n.max(1) }
}
fn bytes(r: &mut Rng, n: usize) -> Vec<u8> { (0..n).map(|_| r.next() as u8).collect() }

/// Random GFX10 programs: plausible opcode words (SOP*, VOP*, SMEM, branches) ending in s_endpgm.
fn program(r: &mut Rng) -> Vec<u8> {
    let n = 1 + r.below(40) as usize;
    let mut w: Vec<u32> = Vec::new();
    for _ in 0..n {
        let kind = r.below(10);
        let x = r.next() as u32;
        let word = match kind {
            0 => 0xBE80_0000 | (x & 0x00FF_FFFF) & !0x0000_0000,          // SOP1
            1 => 0x8000_0000 | (x & 0x3FFF_FFFF),                          // SOP2
            2 => 0xBF80_0000 | (x & 0x007F_FFFF),                          // SOPP (branches, waits)
            3 => 0xBF00_0000 | (x & 0x007F_FFFF),                          // SOPC
            4 => x & 0x7FFF_FFFF,                                          // VOP2
            5 => 0x7E00_0000 | (x & 0x01FF_FFFF),                          // VOP1
            6 => 0x7C00_0000 | (x & 0x01FF_FFFF),                          // VOPC
            7 => { w.push(0xD400_0000 | (x & 0x03FF_FFFF)); r.next() as u32 } // VOP3 (2 words)
            8 => { w.push(0xF400_0000 | (x & 0x03FF_FFFF)); r.next() as u32 } // SMEM
            _ => x,
        };
        w.push(word);
        if r.below(8) == 0 { w.push(r.next() as u32); } // literal
    }
    w.push(0xBF81_0000); // s_endpgm
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let Ok(list) = victory_shader::gfx10::decode(&bytes) else { return vec![0x00, 0x00, 0x81, 0xBF] };
    let mut out = Vec::new();
    for i in list { if i.name().is_some() { for word in &i.words { out.extend_from_slice(&word.to_le_bytes()); } } }
    if out.is_empty() { out.extend_from_slice(&0xBF81_0000u32.to_le_bytes()); }
    out
}


/// Programs that look like compiled shaders: VALU work over v1..v8 and
/// s0..s3, compares into vcc or s16..s30, selects, lane branches (saveexec /
/// execz / restore), ending in stores of v1..v4.
fn structured(r: &mut Rng) -> Vec<u32> {
    let vsrc = |r: &mut Rng| -> u32 {
        match r.below(6) { 0 => r.below(4) as u32, 1 => 128 + r.below(10) as u32, 2 => 240 + r.below(8) as u32, _ => 256 + 1 + r.below(8) as u32 }
    };
    let vreg = |r: &mut Rng| 1 + r.below(8) as u32;
    let mut w: Vec<u32> = Vec::new();
    let ok = |ws: &[u32]| -> bool { let b: Vec<u8> = ws.iter().flat_map(|x| x.to_le_bytes()).collect(); victory_shader::gfx10::decode(&b).is_ok_and(|l| l.len() == 1 && l[0].name().is_some() && l[0].words.len() * 4 == b.len()) };
    let mut open: Vec<(usize, u32)> = Vec::new(); // (branch word index, saved sgpr)
    let n = 1 + r.below(30);
    let mut tries = 0;
    for _ in 0..n {
        let before = w.len();
        let pick = r.below(12);
        match pick {
            0..=2 => { let src1 = vreg(r); w.push((r.below(0x3F) as u32) << 25 | vreg(r) << 17 | src1 << 9 | vsrc(r)); }
            3..=4 => w.push(0x7E00_0000 | vreg(r) << 17 | (r.below(0x80) as u32) << 9 | vsrc(r)),
            5..=6 => { let dst = if r.below(2) == 0 { 106 } else { 16 + 2 * r.below(7) as u32 };
                       w.push(0xD400_0000 | (r.below(0x380) as u32) << 16 | if r.below(3) == 0 { dst } else { vreg(r) });
                       let src2 = if r.below(3) == 0 { 106 } else { vsrc(r) };
                       w.push(vsrc(r) | vsrc(r) << 9 | src2 << 18); }
            7 => w.push(0x7C00_0000 | (r.below(0xFF) as u32) << 17 | vreg(r) << 9 | vsrc(r)),
            8 => { // v_cndmask_b32 by vcc
                   let src1 = vreg(r); w.push(0x0100_0000 & 0 | 1 << 25 | vreg(r) << 17 | src1 << 9 | vsrc(r)); }
            9 if open.len() < 3 => { // if (vcc) {: s_and_saveexec_b32 sN, vcc_lo; s_cbranch_execz
                   let save = 32 + 2 * open.len() as u32;
                   w.push(0xBE80_3C00 | save << 16 | 106); open.push((w.len(), save)); w.push(0xBF88_0000); }
            10 if !open.is_empty() => { // } : s_mov_b32 exec_lo, sN
                   let (at, save) = open.pop().unwrap(); w.push(0xBE80_0300 | 126 << 16 | save);
                   let skip = (w.len() - at - 1 - 1) as u32; w[at] = 0xBF88_0000 | skip; }
            _ => { // sop2 on s0..s3 into s20..s30
                   w.push(0x8000_0000 | (r.below(0x36) as u32) << 23 | (20 + r.below(10) as u32) << 16 | (r.below(4) as u32) << 8 | r.below(4) as u32); }
        }
        if !matches!(pick, 9 | 10) && w.len() > before && !ok(&w[before..]) { w.truncate(before); tries += 1; if tries < 2000 { continue; } }
    }
    while let Some((at, save)) = open.pop() { w.push(0xBE80_0300 | 126 << 16 | save); let skip = (w.len() - at - 1 - 1) as u32; w[at] = 0xBF88_0000 | skip; }
    for k in 0..4u32 { w.push(0x38 << 26 | 0x1C << 18 | 4 * k); w.push(0x80 << 24 | 2 << 16 | (1 + k) << 8); }
    w.push(0xBF81_0000);
    w
}

fn run(name: &str, iters: u64, seed: u64, mut f: impl FnMut(&mut Rng) -> Option<Vec<u8>>) {
    let mut r = Rng(seed);
    let mut found = 0;
    for i in 0..iters {
        let mut snapshot = Rng(r.0);
        let res = catch_unwind(AssertUnwindSafe(|| f(&mut r)));
        if res.is_err() {
            found += 1;
            let input = f_input(&mut snapshot);
            let path = format!("crash-{name}-{i}.bin");
            std::fs::write(&path, input).ok();
            eprintln!("PANIC in {name} at iter {i} (seed state saved {path})");
            if found >= 1 { std::fs::copy(if name.starts_with("emit") { "last-emit.bin" } else { "last-ir.bin" }, &path).ok(); break; }
        }
    }
    eprintln!("{name}: {iters} iterations, {found} panics");
}
fn f_input(_r: &mut Rng) -> Vec<u8> { Vec::new() }

fn show(path: &str) {
    let p = std::fs::read(path).unwrap();
    for i in victory_shader::gfx10::decode(&p).unwrap() { println!("{:?} {:x?}", i.name(), i); }
}
thread_local! { static TALLY: std::cell::RefCell<std::collections::HashMap<String, u64>> = Default::default(); }
fn tally(k: &str) { let k = if k.starts_with("GFX10 code at") { k.split_once(": ").map_or(k, |x| x.1) } else { k }; let k: String = k.chars().take(90).collect(); TALLY.with(|t| *t.borrow_mut().entry(k).or_default() += 1); }
/// Checks emitted DXIL with dxil.dll (`VICTORY_DXIL` or the SDK's) on
/// Windows; a rejected program is saved as `invalid-<n>.bin`.
fn check_dxil(dxil: &[u8], code: &[u8]) {
    #[cfg(windows)]
    {
        use victory_shader::emit::validator::Validator;
        thread_local! { static V: std::cell::RefCell<Option<Validator>> = std::cell::RefCell::new(Validator::load(Validator::default_path().as_deref()).ok()); }
        V.with(|v| {
            let Some(v) = v.borrow_mut().as_mut().map(|v| v.validate(&mut dxil.to_vec())) else { return };
            match v {
                Ok(()) => tally("valid DXIL"),
                Err(e) => {
                    let first = e.lines().find(|l| l.contains("error")).unwrap_or(e.lines().next().unwrap_or("")).to_string();
                    tally(&format!("INVALID DXIL: {first}"));
                    let n = TALLY.with(|t| t.borrow().iter().filter(|(k, _)| k.starts_with("INVALID")).map(|(_, n)| *n).sum::<u64>());
                    if n <= 200 { std::fs::write(format!("invalid-{n}.bin"), code).ok(); }
                }
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (dxil, code);
}
fn main() {
    if std::env::args().nth(1).as_deref() == Some("show") { show(&std::env::args().nth(2).unwrap()); return; }
    let target = std::env::args().nth(1).unwrap_or_default();
    let iters: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let seed = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    std::panic::set_hook(Box::new(|info| {
        let loc = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let msg = info.payload().downcast_ref::<String>().cloned()
            .or_else(|| info.payload().downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
        eprintln!("  panic at {loc}: {msg}");
    }));
    match target.as_str() {
        "decode" => run("decode", iters, seed, |r| { let p = program(r); let _ = victory_shader::gfx10::decode(&p); None }),
        "ir" => run("ir", iters, seed, |r| {
            let p = program(r);
            std::fs::write("last-ir.bin", &p).ok();
            let wave64 = r.below(2) == 0;
            if let Ok(prog) = victory_shader::ir::build(&p, wave64, 256) {
                let _ = victory_shader::structure::structure(&prog);
            }
            None
        }),
        "dxbc" => run("dxbc", iters, seed, |r| {
            let n = r.below(600) as usize;
            let mut d = bytes(r, n);
            if d.len() >= 4 && r.below(2) == 0 { d[..4].copy_from_slice(b"DXBC"); }
            if let Ok(c) = victory_shader::Container::parse(&d) {
                for p in &c.parts { let _ = victory_shader::DxilProgram::parse(p.data); let _ = victory_shader::native::native_programs(p.data); }
            }
            None
        }),
        "native" => run("native", iters, seed, |r| {
            let n = r.below(2000) as usize;
            let mut d = bytes(r, n);
            for _ in 0..r.below(4) { let at = r.below(d.len() as u64 + 1) as usize; if at + 4 <= d.len() { d[at..at+4].copy_from_slice(b"XDAT"); } }
            let _ = victory_shader::native::native_programs(&d);
            let _ = victory_shader::DxilProgram::parse(&d);
            None
        }),
        "pe" => run("pe", iters, seed, |r| {
            let n = r.below(4096) as usize;
            let d = bytes(r, n);
            let _ = victory_pe::ImageInfo::parse(&d);
            let _ = victory_pe::implib::ImportLibrary::parse(&d);
            None
        }),
        "xma" => run("xma", iters, seed, |r| {
            let n = r.below(8192) as usize;
            let d = bytes(r, n);
            let _ = victory_formats::xma::packet(&d, r.below(8) as usize);
            let _ = victory_formats::xma::read_frame(&d, r.below(n as u64 * 8 + 1) as usize);
            None
        }),
        "sgd5" => run("sgd5", iters, seed, |r| {
            let n = r.below(4096) as usize;
            let mut d = bytes(r, n);
            if d.len() >= 4 && r.below(2) == 0 { d[..4].copy_from_slice(b"SGD5"); }
            for _rec in victory_game::sgd5::records(&d).take(64) {}
            let _ = victory_game::shaderpack::find(&d);
            let _ = victory_game::shaderpack::program_id(&d);
            None
        }),
        "emit" => {
            use victory_shader::native::*;
            let mut container = b"DXBC".to_vec(); container.extend([0u8; 16]); container.extend(1u16.to_le_bytes()); container.extend(0u16.to_le_bytes()); container.extend(0x20u32.to_le_bytes()); container.extend(0u32.to_le_bytes());
            run("emit", iters, seed, move |r| {
                let body = program(r);
                let mut words: Vec<u32> = vec![0xBE85_0380]; // s_mov_b32 s5, 0
                words.extend([0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]); // s_load_dwordx4 s[8:11], s[4:5]
                for i in 0..4u32 { words.push(0x7E00_0000 | (i + 1) << 17 | 0x001 << 9 | i); }
                let mut code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                code.extend_from_slice(&body);
                if let Ok(path) = std::env::var("REPLAY") { code = std::fs::read(path).unwrap(); }
                std::fs::write("last-emit.bin", &code).ok();
                let wave64 = r.below(2) == 0;
                let threads = [1u32, 32, 64, 128][r.below(4) as usize];
                let native = NativeProgram {
                    stage: HwStage::Cs, wave64, vgprs: 64,
                    layout: vec![
                        RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
                        RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
                    ],
                    used: vec![UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords: 4, user_data: 4, slot: 1 }],
                    io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1 | (r.below(4) as u32) << 15, threads: [threads, 1, 1] }),
                    code,
                };
                let c = victory_shader::Container::parse(&container).unwrap();
                let Ok(p) = victory_shader::ir::build(&native.code, native.wave64, native.vgprs) else { return None };
                let st = victory_shader::structure::structure(&p);
                let Ok(res) = victory_shader::resource::track(&p, &native.layout, native.stage, st.as_ref().ok()) else { return None };
                let Ok(io) = victory_shader::io::raise(&p, &native) else { return None };
                let Ok(st) = st else { return None };
                if let Ok(sh) = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res) { check_dxil(&sh.dxil, &native.code) }
                None
            })
        }
        "emit2" => {
            use victory_shader::native::*;
            let mut container = b"DXBC".to_vec(); container.extend([0u8; 16]); container.extend(1u16.to_le_bytes()); container.extend(0u16.to_le_bytes()); container.extend(0x20u32.to_le_bytes()); container.extend(0u32.to_le_bytes());
            run("emit2", iters, seed, move |r| {
                let body: Vec<u8> = structured(r).iter().flat_map(|w| w.to_le_bytes()).collect();
                let mut words: Vec<u32> = vec![0xBE85_0380]; // s_mov_b32 s5, 0
                words.extend([0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]); // s_load_dwordx4 s[8:11], s[4:5]
                for i in 0..4u32 { words.push(0x7E00_0000 | (i + 1) << 17 | 0x001 << 9 | i); }
                let mut code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                code.extend_from_slice(&body);
                if let Ok(path) = std::env::var("REPLAY") { code = std::fs::read(path).unwrap(); }
                std::fs::write("last-emit.bin", &code).ok();
                let wave64 = r.below(2) == 0;
                let threads = [1u32, 32, 64, 128][r.below(4) as usize];
                let native = NativeProgram {
                    stage: HwStage::Cs, wave64, vgprs: 64,
                    layout: vec![
                        RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
                        RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
                    ],
                    used: vec![UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords: 4, user_data: 4, slot: 1 }],
                    io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1 | (r.below(4) as u32) << 15, threads: [threads, 1, 1] }),
                    code,
                };
                let c = victory_shader::Container::parse(&container).unwrap();
                let Ok(p) = victory_shader::ir::build(&native.code, native.wave64, native.vgprs) else { tally("ir"); return None };
                let st = victory_shader::structure::structure(&p);
                let Ok(res) = victory_shader::resource::track(&p, &native.layout, native.stage, st.as_ref().ok()) else { tally("res"); return None };
                let Ok(io) = victory_shader::io::raise(&p, &native) else { return None };
                let Ok(st) = st else { tally("structure"); return None };
                match victory_shader::emit::emit(&c, &native, &p, &st, &io, &res) { Ok(sh) => { tally("ok"); check_dxil(&sh.dxil, &native.code) } Err(e) => tally(&format!("{e}")) }
                None
            })
        }
        _ => eprintln!("usage: vfuzz <decode|ir|dxbc|native|pe|xma|sgd5|emit|emit2> [iterations] [seed] | vfuzz show <file>"),
    }
    TALLY.with(|t| { let mut v: Vec<_> = t.borrow().iter().map(|(k, n)| (*n, k.clone())).collect(); v.sort_by(|a, b| b.cmp(a)); for (n, k) in v.iter().take(25) { eprintln!("  {n:>8} {k}"); } });
}
