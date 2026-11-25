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
        let pick = r.below(16);
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
            12 => { // buffer_load_dword vD, vA, s[8:11], 0 offen offset:k (vA masked to 0..1020 first)
                   let (a, d) = (vreg(r), vreg(r));
                   w.push(0x3600_00FF | 0x1B << 25 | a << 17 | a << 9); w.push(0x3FC); // v_and_b32 vA, 0x3fc, vA
                   w.push(0x38 << 26 | 0x0C << 18 | 1 << 12 | 4 * r.below(16) as u32); w.push(0x80 << 24 | 2 << 16 | d << 8 | a); }
            13 => { // ds_write_b32 / ds_add_u32 at a masked address
                   let (a, d) = (vreg(r), vreg(r));
                   w.push(0x3600_00FF | 0x1B << 25 | a << 17 | a << 9); w.push(0x3FC);
                   let op = if r.below(2) == 0 { 0x0D } else { 0x00 };
                   w.push(0x36 << 26 | op << 18 | r.below(16) as u32 * 4); w.push(d << 8 | a); }
            14 => { // ds_read_b32 at a masked address
                   let (a, d) = (vreg(r), vreg(r));
                   w.push(0x3600_00FF | 0x1B << 25 | a << 17 | a << 9); w.push(0x3FC);
                   w.push(0x36 << 26 | 0x36 << 18 | r.below(16) as u32 * 4); w.push(d << 24 | a); }
            15 if open.is_empty() => w.push(0xBF8A_0000), // s_barrier, outside lane branches
            _ => { // sop2 on s0..s3 into s20..s30
                   w.push(0x8000_0000 | (r.below(0x36) as u32) << 23 | (20 + r.below(10) as u32) << 16 | (r.below(4) as u32) << 8 | r.below(4) as u32); }
        }
        if !matches!(pick, 9 | 10 | 12 | 13 | 14) && w.len() > before && !ok(&w[before..]) { w.truncate(before); tries += 1; if tries < 2000 { continue; } }
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
                    if n <= 200 { std::fs::write(format!("invalid-{}-{n}.bin", std::process::id()), code).ok(); }
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
        "texture" => run("texture", iters, seed, |r| {
            use victory_formats::texture::*;
            let side = |r: &mut Rng| match r.below(8) { 0 => r.next() as u32, 1 => 1 + r.below(16384) as u32, _ => 1 + r.below(300) as u32 };
            let desc = TextureDesc {
                dimension: [Dimension::Tex1d, Dimension::Tex2d, Dimension::Tex3d][r.below(3) as usize],
                width: side(r), height: side(r), depth_or_array: if r.below(4) == 0 { side(r) } else { 1 + r.below(6) as u32 },
                mips: r.below(17) as u32,
                format: if r.below(4) == 0 { r.below(135) as u32 } else { [28, 87, 71, 77, 98, 10, 2, 41, 40, 45, 20, 61, 49, 24, 16, 95, 80, 83, 55, 56][r.below(20) as usize] },
                samples: [1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 3, 0][r.below(12) as usize],
                swizzle: Swizzle(if r.below(4) == 0 { r.below(32) as u32 } else { [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 23, 24, 25, 26, 27][r.below(16) as usize] }),
            };
            std::fs::write("last-texture.txt", format!("{desc:?}")).ok();
            let Ok(t) = Texture::new(&desc) else { tally("texture refused"); return None };
            tally("texture ok");
            for p in &t.planes {
                let l = &p.layout;
                for _ in 0..4 {
                    let mip = r.below(l.mips.len() as u64 + 1) as u32;
                    let (x, y, z) = (r.below(300) as u32, r.below(300) as u32, r.below(8) as u32);
                    let _ = l.offset(mip, x, y, z);
                    let _ = l.bytes_from(mip);
                    if l.size > 8 << 20 || mip as usize >= l.mips.len() { continue; }
                    let m = l.subresource_depth(mip);
                    let tiled = vec![0u8; l.bytes_from(mip) as usize];
                    let (w, h) = ((desc.width >> mip).max(1) as usize, (desc.height >> mip).max(1) as usize);
                    let row = w * l.desc.element_bytes as usize + r.below(64) as usize;
                    let depth_pitch = row * h;
                    let need = depth_pitch * m.max(1) as usize;
                    if need > 32 << 20 { continue; }
                    let mut linear = vec![0u8; need];
                    let slice = r.below(4) as u32;
                    tally(if l.detile(&tiled, mip, slice, &mut linear, row, depth_pitch).is_ok() { "detile ok" } else { "detile refused" });
                    // Round trip: tiling distinct data and detiling it gives it back, so
                    // no two elements of a subresource share tiled bytes.
                    let src: Vec<u8> = (0..need).map(|_| r.next() as u8).collect();
                    let mut tiled = tiled;
                    if l.tile(&mut tiled, mip, slice, &src, row, depth_pitch).is_ok() {
                        let mut back = vec![0u8; need];
                        l.detile(&tiled, mip, slice, &mut back, row, depth_pitch).expect("detile after tile");
                        let bytes = l.mips[mip as usize].width as usize * l.desc.element_bytes as usize;
                        let rows = l.mips[mip as usize].height as usize;
                        for d in 0..m.max(1) as usize { for y in 0..rows {
                            let at = d * depth_pitch + y * row;
                            assert_eq!(&back[at..at + bytes], &src[at..at + bytes], "round trip differs: mip {mip} slice {slice} depth {d} row {y}: {:?}", l.desc);
                        } }
                        tally("round trip ok");
                    }
                }
            }
            // Two subresources never share tiled bytes: tile one, then the other,
            // and the first still detiles to what it was given.
            for p in &t.planes {
                let l = &p.layout;
                if l.size > 8 << 20 || l.mips.is_empty() { continue; }
                let slices = if desc.dimension == Dimension::Tex3d { 1 } else { l.slices };
                let pick = |r: &mut Rng| (r.below(l.mips.len() as u64) as u32, r.below(u64::from(slices)) as u32);
                let (a, b) = (pick(r), pick(r));
                if a == b { continue; }
                let mut tiled = vec![0u8; l.size as usize];
                let size = |mip: u32| {
                    let m = &l.mips[mip as usize];
                    let row = m.width as usize * l.desc.element_bytes as usize;
                    (row, row * m.height as usize, row * m.height as usize * m.depth.max(1) as usize)
                };
                let (ra, da, na) = size(a.0);
                let (rb, db, nb) = size(b.0);
                if na.max(nb) > 32 << 20 { continue; }
                let src_a: Vec<u8> = (0..na).map(|_| r.next() as u8).collect();
                let src_b: Vec<u8> = (0..nb).map(|_| r.next() as u8).collect();
                if l.tile(&mut tiled, a.0, a.1, &src_a, ra, da).is_err() || l.tile(&mut tiled, b.0, b.1, &src_b, rb, db).is_err() { continue; }
                let mut back = vec![0u8; na];
                l.detile(&tiled, a.0, a.1, &mut back, ra, da).expect("detile");
                assert!(back == src_a, "subresources {a:?} and {b:?} overlap: {:?}", l.desc);
                tally("no overlap");
            }
            None
        }),
        "bcpack" => {
            let tables = victory_formats::bcpack::Tables::builtin().expect("BCPack tables (run Initialize-FormatTables.ps1)");
            run("bcpack", iters, seed, move |r| {
                let bc = [1u8, 3, 4, 5, 6, 7, 2][r.below(7) as usize];
                let n = r.below(4096) as usize;
                let src = bytes(r, n);
                let mut dst = vec![0u8; 16 * r.below(512) as usize];
                let _ = victory_formats::bcpack::decode(bc, &src, &mut dst, &tables);
                None
            })
        }
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
                    io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1 | 2 << 15, threads: [threads, 1, 1] }),
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
        _ => eprintln!("usage: vfuzz <decode|ir|dxbc|native|pe|xma|sgd5|texture|bcpack|emit|emit2> [iterations] [seed] | vfuzz show <file>"),
    }
    TALLY.with(|t| { let mut v: Vec<_> = t.borrow().iter().map(|(k, n)| (*n, k.clone())).collect(); v.sort_by(|a, b| b.cmp(a)); for (n, k) in v.iter().take(25) { eprintln!("  {n:>8} {k}"); } });
}
