"""Prepare DSSAT FileX from the deposited author-supplied templates.

Offline, non-executing dry-run only. The user must supply the rest of the DSSAT
weather/soil/cultivar files and DSSAT executable to perform real simulations.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "Maize": ("TPL_MAIZE.MZX", "CHXF2401.MZX"),
    "Soybean": ("TPL_SOY.SBX", "CHXF2402.SBX"),
    "Rice": ("TPL_RICE.RIX", "CHXF2403.RIX"),
}
TOKENS = (b"{SOIL_ID} ", b"{POP} {POE}", b"{FN}", b"{FP}", b"{FK}")


def convert_fertilizer(weight: float):
    # Matches supplied 15-15-15 batch runner's numeric conversions.
    amount=int(round(weight))
    n=int(round(amount * 0.15))
    p=int(round(amount * 0.15 * 0.436))
    k=int(round(amount * 0.15 * 0.830))
    return n,p,k


def render(template: bytes, *, crop: str, soil_id: str, density: float, fertilizer: float):
    if crop not in TEMPLATES: raise ValueError(f"Unknown crop: {crop}")
    if not (soil_id.startswith('XY') and len(soil_id)==10 and soil_id[2:].isdigit()):
        raise ValueError("Soil_ID must have format XY00000001 (10 ASCII chars)")
    if not (0 < density < 1000 and 0 <= fertilizer <= 10000):
        raise ValueError("Density/fertilizer outside minimal sanity ranges")
    pop=f"{density:>5.1f}".encode('ascii')
    if len(pop)!=5: raise ValueError("DSSAT PPOP/PPOE width overflow")
    n,p,k=convert_fertilizer(fertilizer)
    fields={
        b"{SOIL_ID} ":soil_id.encode('ascii'),
        b"{POP} {POE}":pop+b' '+pop,
        b"{FN}":f"{n:>4d}".encode('ascii'),
        b"{FP}":f"{p:>4d}".encode('ascii'),
        b"{FK}":f"{k:>4d}".encode('ascii'),
    }
    for old,new in fields.items():
        if template.count(old)!=1: raise ValueError(f"Expected one token {old!r}, got {template.count(old)}")
        if len(old)!=len(new): raise ValueError(f"Fixed-width misalignment for {old!r}")
        template=template.replace(old,new)
    if b"{" in template or b"}" in template: raise ValueError("Unsubstituted template token")
    return template


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--design',type=Path,default=ROOT/'dssat/example_design/archived_rows_for_template_test.csv')
    ap.add_argument('--out',type=Path,default=ROOT/'results/generated_dssat_examples')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    seen=set()
    with args.design.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    for index,row in enumerate(rows,1):
        crop=row['Crop']
        tpl_name,out_name=TEMPLATES[crop]
        template=(ROOT/'dssat/templates'/tpl_name).read_bytes()
        output=render(template,crop=crop,soil_id=row['Soil_ID'],
                      density=float(row['Density_plants_m2']),
                      fertilizer=float(row['Fert_Weight_kg_ha']))
        target=args.out/(f"sample_{index:04d}_"+out_name)
        if target.name in seen:raise ValueError("Duplicate output filename")
        seen.add(target.name)
        target.write_bytes(output)
        print(f"Prepared {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")
    print("Offline template check complete; DSSAT itself was NOT run.")

if __name__=='__main__': main()
