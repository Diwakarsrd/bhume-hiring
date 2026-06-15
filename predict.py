#!/usr/bin/env python3
"""
BhuMe boundary correction — predict.py  (v2: IDW + NCC refinement)
===================================================================

Method: IDW spatial interpolation + image-based NCC residual refinement
-----------------------------------------------------------------------

STAGE 1 — IDW from example truths (primary correction)
  Measured offsets at truth control points, interpolated to every plot via
  Inverse Distance Weighting (power=2). Reliable everywhere; accuracy
  degrades gracefully with distance from truths.

STAGE 2 — NCC residual refinement (image-based correction)
  For each plot, render its boundary edges as a binary mask at the IDW-predicted
  position, then search ±residual_px in the boundaries.tif raster for the
  shift that maximises Normalised Cross-Correlation (NCC via FFT).
  The NCC peak sharpness (best − mean of top-10%) measures how clear the
  image evidence is. Only apply the NCC correction when sharpness is high.

CONFIDENCE signals (four inputs, combined):
  1. IDW dist_score  — exponential decay with distance to nearest truth
  2. IDW consistency — variance of nearby truth shifts (low = consistent)
  3. NCC sharpness   — applied correction has clear image support
  4. NCC–IDW agree   — two independent signals produce similar answers
  Combined via geometric mean, clipped to [0.30, 0.88].

DECISION per plot:
  - sharp NCC (≥15) AND agrees with IDW (≤8m): IDW+NCC average, boost conf
  - sharp NCC but disagrees with IDW: use IDW only, penalise conf
  - weak NCC (<15): use IDW only, confidence from distance/consistency
  - no truths, no raster: flag

Usage:
    python predict.py <village_dir>
    Reads: input.geojson, example_truths.geojson, boundaries.tif
    Writes: predictions.geojson

Requirements: Python 3.9+, numpy, scipy  (no geopandas/rasterio needed)
"""

from __future__ import annotations
import json, math, struct, sys, copy
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve


# ── TIF reader (pure Python + numpy) ─────────────────────────────────────────

def read_tiff(path):
    with open(path, 'rb') as f: raw = f.read()
    end = '<' if raw[:2] == b'II' else '>'
    ifd = struct.unpack_from(end+'I', raw, 4)[0]
    n = struct.unpack_from(end+'H', raw, ifd)[0]
    tags = {}
    for i in range(n):
        eo = ifd+2+i*12
        tag = struct.unpack_from(end+'H', raw, eo)[0]
        typ = struct.unpack_from(end+'H', raw, eo+2)[0]
        cnt = struct.unpack_from(end+'I', raw, eo+4)[0]
        vo = eo+8
        tsz = {1:1,2:1,3:2,4:4,5:8,12:8}.get(typ,4)
        if cnt*tsz <= 4: vraw = raw[vo:vo+cnt*tsz]
        else:
            ptr = struct.unpack_from(end+'I', raw, vo)[0]; vraw = raw[ptr:ptr+cnt*tsz]
        if   typ==3:  vals = [struct.unpack_from(end+'H', vraw, j*2)[0] for j in range(cnt)]
        elif typ==4:  vals = [struct.unpack_from(end+'I', vraw, j*4)[0] for j in range(cnt)]
        elif typ==12: vals = [struct.unpack_from(end+'d', vraw, j*8)[0] for j in range(cnt)]
        else:         vals = list(vraw)
        tags[tag] = vals
    w=tags[256][0]; h=tags[257][0]
    sx=tags[33550][0]; sy=tags[33550][1]
    ox=tags[33922][3]; oy=tags[33922][4]
    data = bytearray()
    for off,bc in zip(tags[273], tags[279]): data.extend(raw[off:off+bc])
    arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(h, w)
    return arr, sx, sy, ox, oy


# ── Coordinate helpers ────────────────────────────────────────────────────────

def lonlat_to_merc(lon, lat):
    x = lon * 20037508.342 / 180
    y = math.log(math.tan((90+lat)*math.pi/360)) / math.pi * 20037508.342
    return x, y

def centroid_lonlat(geom):
    ring = geom['coordinates'][0][0] if geom['type']=='MultiPolygon' else geom['coordinates'][0]
    n = len(ring)-1
    return sum(c[0] for c in ring[:n])/n, sum(c[1] for c in ring[:n])/n

def haversine_m(lon1, lat1, lon2, lat2):
    R=6_371_000; phi1,phi2=math.radians(lat1),math.radians(lat2)
    dphi=math.radians(lat2-lat1); dlam=math.radians(lon2-lon1)
    a=math.sin(dphi/2)**2+math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2*R*math.asin(math.sqrt(min(1.0, a)))

def metres_to_deg(dx_m, dy_m, ref_lat):
    lm = 111320*math.cos(math.radians(ref_lat))
    return dx_m/lm, dy_m/111320

def deg_to_metres(dlon, dlat, ref_lat):
    return dlon*111320*math.cos(math.radians(ref_lat)), dlat*111320

def shift_geometry(geom, dlon, dlat):
    g = copy.deepcopy(geom)
    def sc(c):
        if isinstance(c[0], (int,float)): return [c[0]+dlon, c[1]+dlat]+list(c[2:])
        return [sc(x) for x in c]
    g['coordinates'] = sc(g['coordinates']); return g


# ── IDW ───────────────────────────────────────────────────────────────────────

def idw_shift(lon, lat, shifts, power=2.0):
    dists = [max(1.0, haversine_m(lon, lat, s['lon'], s['lat'])) for s in shifts]
    md = min(dists)
    if md <= 2.0:
        idx = dists.index(md); return shifts[idx]['dx_m'], shifts[idx]['dy_m'], md
    ws = [1/d**power for d in dists]; ws_ = sum(ws)
    return (sum(w*s['dx_m'] for w,s in zip(ws,shifts))/ws_,
            sum(w*s['dy_m'] for w,s in zip(ws,shifts))/ws_, md)

def idw_confidence(lon, lat, min_dist_m, shifts):
    dist_score = math.exp(-math.log(2)*min_dist_m/1500)
    dist_score = max(0.05, min(1.0, dist_score))
    nearby = [(haversine_m(lon,lat,s['lon'],s['lat']), s['dx_m'], s['dy_m'])
              for s in shifts if haversine_m(lon,lat,s['lon'],s['lat']) <= 2500]
    if len(nearby) < 2:
        con = 0.6
    else:
        ws = [1/max(d,1) for d,_,_ in nearby]; ws_ = sum(ws)
        mdx = sum(w*dx for w,(_,dx,__) in zip(ws,nearby))/ws_
        mdy = sum(w*dy for w,(_,__,dy) in zip(ws,nearby))/ws_
        vdx = sum(w*(dx-mdx)**2 for w,(_,dx,__) in zip(ws,nearby))/ws_
        vdy = sum(w*(dy-mdy)**2 for w,(_,__,dy) in zip(ws,nearby))/ws_
        con = math.exp(-math.sqrt((vdx+vdy)/2)/15)
        con = max(0.1, min(1.0, con))
    return round(max(0.30, min(0.85, math.sqrt(dist_score*con))), 3)


# ── Edge rasteriser ───────────────────────────────────────────────────────────

def render_edges(geom, ox, oy, sx, sy, idw_dx_m, idw_dy_m, pad=12, thickness=1):
    """Render polygon edges shifted by (idw_dx_m, idw_dy_m) into raster pixel space."""
    ring = geom['coordinates'][0][0] if geom['type']=='MultiPolygon' else geom['coordinates'][0]
    dpx = idw_dx_m / sx; dpy = -idw_dy_m / sy
    mercs = [lonlat_to_merc(c[0], c[1]) for c in ring]
    cols  = [(m[0]-ox)/sx + dpx for m in mercs]
    rows_ = [(oy-m[1])/sy + dpy for m in mercs]
    r0=int(min(rows_))-pad; c0=int(min(cols))-pad
    r1=int(max(rows_))+pad; c1=int(max(cols))+pad
    H=max(r1-r0, 1); W=max(c1-c0, 1)
    mask = np.zeros((H, W), dtype=np.float32)
    pts = [(c-c0, r-r0) for c,r in zip(cols, rows_)]
    for i in range(len(pts)-1):
        x0,y0 = pts[i]; x1,y1 = pts[i+1]
        steps = int(max(abs(x1-x0), abs(y1-y0)))+1
        xs = np.round(np.linspace(x0, x1, steps+1)).astype(int)
        ys = np.round(np.linspace(y0, y1, steps+1)).astype(int)
        for t in range(-thickness, thickness+1):
            rr = np.clip(ys+t, 0, H-1); cc = np.clip(xs+t, 0, W-1)
            mask[rr, cc] = 1.0
    return mask, max(0, r0), max(0, c0)


# ── NCC residual refinement ───────────────────────────────────────────────────

SHARP_THRESHOLD  = 15.0   # min NCC sharpness to trust image evidence
AGREE_THRESHOLD  = 8.0    # metres — NCC and IDW "agree" within this
RESIDUAL_PX      = 10     # search ±pixels around IDW position
NCC_PAD          = 12     # pixels of padding around plot bbox


def ncc_refine(mask, bnd_arr, tmpl_r0, tmpl_c0, sx, sy):
    """
    Search ±RESIDUAL_PX around the IDW-predicted template position.
    Returns (residual_dx_m, residual_dy_m, sharpness).
    """
    H, W = mask.shape
    margin = RESIDUAL_PX
    sr0 = max(0, tmpl_r0-margin); sr1 = min(bnd_arr.shape[0], tmpl_r0+H+margin)
    sc0 = max(0, tmpl_c0-margin); sc1 = min(bnd_arr.shape[1], tmpl_c0+W+margin)
    if sr1-sr0 < H or sc1-sc0 < W:
        return 0.0, 0.0, 0.0

    search = bnd_arr[sr0:sr1, sc0:sc1].astype(np.float32)
    t = mask - mask.mean(); ts = mask.std()
    if ts < 1e-6: return 0.0, 0.0, 0.0
    t /= ts
    corr = fftconvolve(search, t[::-1,::-1], mode='full')
    corr /= mask.size

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[peak_idx])

    # Convert to shift relative to IDW position
    dr = peak_idx[0] - (H-1) - (tmpl_r0 - sr0)
    dc = peak_idx[1] - (W-1) - (tmpl_c0 - sc0)

    flat = corr.flatten(); flat.sort()
    top10 = flat[-max(1, len(flat)//10):]
    sharpness = float(peak_val - top10.mean())

    return float(dc*sx), float(-dr*sy), sharpness   # dx_m, dy_m, sharpness


# ── Main ──────────────────────────────────────────────────────────────────────

def predict_village(village_dir):
    d = Path(village_dir)
    input_path  = d/'input.geojson'
    truths_path = d/'example_truths.geojson'
    bnd_path    = d/'boundaries.tif'
    out_path    = d/'predictions.geojson'

    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}")

    with open(input_path) as f: features = json.load(f)['features']
    print(f"Loaded {len(features)} plots from {input_path.name}")

    # ── IDW control points ────────────────────────────────────────────────────
    shifts = []
    if truths_path.exists():
        with open(truths_path) as f: truth_feats = json.load(f)['features']
        inp_idx = {str(f['properties']['plot_number']): f for f in features}
        for feat in truth_feats:
            pn = str(feat['properties']['plot_number'])
            if pn not in inp_idx: continue
            tc = centroid_lonlat(feat['geometry'])
            ic = centroid_lonlat(inp_idx[pn]['geometry'])
            dx, dy = deg_to_metres(tc[0]-ic[0], tc[1]-ic[1], (tc[1]+ic[1])/2)
            shifts.append({'plot_number':pn,'lon':ic[0],'lat':ic[1],'dx_m':dx,'dy_m':dy})
        dxs = [s['dx_m'] for s in shifts]; dys = [s['dy_m'] for s in shifts]
        print(f"IDW: {len(shifts)} truths | dx=[{min(dxs):.0f},{max(dxs):.0f}]m dy=[{min(dys):.0f},{max(dys):.0f}]m")

    # ── Boundaries raster ─────────────────────────────────────────────────────
    bnd_arr = None
    if bnd_path.exists():
        try:
            bnd_arr, bnd_sx, bnd_sy, bnd_ox, bnd_oy = read_tiff(bnd_path)
            print(f"NCC: boundaries raster {bnd_arr.shape}, px={bnd_sx:.2f}m")
        except Exception as e:
            print(f"WARNING: could not load boundaries.tif — NCC disabled ({e})")

    # ── Per-plot prediction ───────────────────────────────────────────────────
    stats = dict(ncc_agree=0, ncc_disagree=0, idw_only=0, flagged=0)
    out_features = []

    for feat in features:
        pn   = str(feat['properties']['plot_number'])
        geom = feat['geometry']
        clon, clat = centroid_lonlat(geom)

        if not shifts and bnd_arr is None:
            out_features.append({'type':'Feature','properties':{
                'plot_number':pn,'status':'flagged',
                'method_note':'No truths and no boundaries raster'},'geometry':geom})
            stats['flagged'] += 1; continue

        # Stage 1 — IDW
        if shifts:
            idw_dx, idw_dy, idw_dist = idw_shift(clon, clat, shifts)
            idw_conf = idw_confidence(clon, clat, idw_dist, shifts)
        else:
            idw_dx = idw_dy = 0.0; idw_conf = 0.30; idw_dist = 9999

        # Stage 2 — NCC residual (search ±RESIDUAL_PX around IDW position)
        ncc_dx = ncc_dy = 0.0; ncc_sharp = 0.0; ncc_ok = False
        if bnd_arr is not None:
            try:
                mask, tmpl_r0, tmpl_c0 = render_edges(
                    geom, bnd_ox, bnd_oy, bnd_sx, bnd_sy,
                    idw_dx, idw_dy, pad=NCC_PAD)
                if mask.shape[0] > 4 and mask.shape[1] > 4:
                    ncc_dx, ncc_dy, ncc_sharp = ncc_refine(
                        mask, bnd_arr, tmpl_r0, tmpl_c0, bnd_sx, bnd_sy)
                    ncc_ok = ncc_sharp >= SHARP_THRESHOLD
            except Exception:
                pass

        # Stage 3 — combine
        agree_dist = math.sqrt(ncc_dx**2 + ncc_dy**2) if ncc_ok else 0.0

        if ncc_ok and agree_dist <= AGREE_THRESHOLD:
            # NCC finds a clear nearby correction that agrees with IDW direction
            final_dx = idw_dx + ncc_dx * 0.5   # blend: IDW + half of NCC residual
            final_dy = idw_dy + ncc_dy * 0.5
            ncc_score = min(0.88, 0.45 + ncc_sharp/100)
            conf = round(min(0.88, math.sqrt(ncc_score * idw_conf) * 1.15), 3)
            method = (f"IDW+NCC: dx={final_dx:+.1f}m dy={final_dy:+.1f}m "
                      f"(idw_conf={idw_conf:.2f} ncc_sharp={ncc_sharp:.1f})")
            stats['ncc_agree'] += 1
        elif ncc_ok and agree_dist > AGREE_THRESHOLD:
            # NCC found something but it's far from IDW — don't trust it
            final_dx = idw_dx; final_dy = idw_dy
            conf = round(max(0.30, idw_conf * 0.80), 3)
            method = (f"IDW only (NCC disagrees {agree_dist:.0f}m): "
                      f"dx={final_dx:+.1f}m dy={final_dy:+.1f}m (conf={conf:.2f})")
            stats['ncc_disagree'] += 1
        else:
            # No reliable NCC — pure IDW
            final_dx = idw_dx; final_dy = idw_dy
            conf = round(idw_conf, 3)
            method = (f"IDW: dx={final_dx:+.1f}m dy={final_dy:+.1f}m "
                      f"(nearest truth {idw_dist:.0f}m conf={conf:.2f})")
            stats['idw_only'] += 1

        dlon, dlat = metres_to_deg(final_dx, final_dy, clat)
        out_features.append({'type':'Feature','properties':{
            'plot_number':pn,'status':'corrected',
            'confidence':conf,'method_note':method},
            'geometry':shift_geometry(geom, dlon, dlat)})

    # ── Write ─────────────────────────────────────────────────────────────────
    out_path.write_text(json.dumps({'type':'FeatureCollection','features':out_features},
                                   separators=(',',':')))
    confs = [f['properties']['confidence'] for f in out_features
             if f['properties']['status']=='corrected']
    print(f"Wrote {out_path}")
    print(f"  NCC+IDW agree: {stats['ncc_agree']}  |  NCC disagree (IDW kept): {stats['ncc_disagree']}")
    print(f"  IDW only: {stats['idw_only']}  |  Flagged: {stats['flagged']}")
    if confs:
        print(f"  Confidence: min={min(confs):.2f}  max={max(confs):.2f}  mean={sum(confs)/len(confs):.3f}")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <village_dir> [<village_dir2> ...]")
        print("  e.g.: python predict.py data/vadnerbhairav")
        sys.exit(1)
    for vd in sys.argv[1:]:
        print(f"\n{'='*60}\nVillage: {vd}\n{'='*60}")
        predict_village(vd)
    print("\nDone. Self-score at https://hiring.bhume.in/test")

if __name__ == "__main__":
    main()
