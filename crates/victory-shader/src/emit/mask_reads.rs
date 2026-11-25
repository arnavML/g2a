//! Instructions that read a lane mask as a number rather than as a lane
//! set: in compute they need the console's lane order.
use super::{Fields, Inst, Operand, Reg, Structured, Value, NULL};

pub(in crate::emit) fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// A `v_cndmask_b32` that picks by a lane mask the code holds as a number
/// (no tracked mask): it reads this lane's bit of that number, so in compute
/// it needs the console's lane order like any other cross-lane read.
pub(super) fn selects_by_number(st: &Structured, i: &Inst) -> bool {
    if i.name() != "v_cndmask_b32" {
        return false;
    }
    let Fields::Valu(v) = &i.fields else { return false };
    match v.src.get(2) {
        Some(&Operand::Scalar { code, .. }) => {
            i.read(Reg::S(code)).is_some_and(|m| !st.masks.contains_key(&m))
        }
        _ => false,
    }
}
