"""IMRI re-evaluation at dt=5: per-channel SNR + overlap/chi2 of the CURRENT best-fit points.

Motivation: at dt=10 the Nyquist frequency is 0.05 Hz.  Prograde IMRIs carry power above
that, which aliases and dumps spurious power into the (otherwise null) T channel -- the
dt=10 run showed T carrying up to 97% of the SNR, while the one dt=5 case and the
short-T tails cases showed T ~ 0.  This script re-evaluates everything at dt=5
(Nyquist 0.1 Hz), leaving all other settings identical.

For each case it reports, at dt=5:
  * SNR_A, SNR_E, SNR_T and the quadrature total of the 1PA injection
  * overlap and chi2 = <r|r> of the stored best-fit points (0PA / PN / simple),
    which were optimised at dt=10 -- so the change tells you how much dt=10 distorted them.

Best-fit points are read from the results_imri_*.json files (best of the available seeds).
Requires SuperKludge_r on the 'hybrid' branch (PN/simple wiring).

Output: snr_overlap_dt5_imri.json + printed tables.
Run:  python snr_overlap_dt5_imri.py
"""
import json, os, glob
from datetime import datetime, timezone
import numpy as np

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product

try:
    import cupy as cp; xp = cp
except ImportError:
    xp = np; print("[INFO] CuPy not found, using NumPy.")

DT_NEW = 5.0                      # <-- the point of this script
F_MIN, NCH, USE_GPU = 1e-5, 3, True
CHAN, NAMES = [A1TDISens, E1TDISens, T1TDISens], ["A", "E", "T"]
P14 = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
       "Phi_phi0","Phi_theta0","Phi_r0"]
P9  = ["m1","m2","a","p0","e0","qS","phiS","Phi_phi0","Phi_r0"]
P11 = P9 + ["dev_1","dev_2"]
SIGNAL_ARRAY = "/scratch/e1583490/SK_files/data/signal/signal_parameter_array_IMRI.npy"
HERE = os.path.dirname(os.path.abspath(__file__))

def pn_tail(a,b):     return [a,b,0.0,0.0]
def simple_tail(a,b): return [0.0,0.0,a,b]
TAIL = {"PN": pn_tail, "simple": simple_tail}

def _f(x): return float(x.get()) if hasattr(x,"get") else float(x)
def fmask(n,dt): return (xp.fft.rfftfreq(n,dt) > F_MIN)[1:]
def hp(w,dt):
    n=w.shape[-1]; f=xp.fft.rfftfreq(n,dt)
    return xp.fft.irfft(xp.fft.rfft(w,axis=-1)*(f>=F_MIN),n=n,axis=-1)

# ---------------- best-fit points from the JSONs ----------------
def _K(fn):
    p=os.path.join(HERE,fn)
    if not os.path.exists(p): return {}
    d=json.load(open(p)); return {c["name"]:c for c in d.get("points",d.get("cases",[]))}

def best_fits():
    """-> {case_name: {model: params}} using the best available seed."""
    out={}
    dv,dvf,dv2 = _K("results_imri_grid_diverse.json"),_K("results_imri_grid_diverse_dev_from_0pa.json"),_K("results_imri_grid_diverse2.json")
    for n,p in dv.items():
        R=p["runs"]; F=dvf.get(n,{}).get("runs",{})
        e={"0PA":R["0PA"]["from_MAP"]["params"]}
        for m in ("PN","simple"):
            c=[R[m]["from_MAP"]]+([F[m]] if m in F else [])
            e[m]=max(c,key=lambda r:r["ov_final"])["params"]
        out["grid_"+n]=e
    for n,p in dv2.items():
        R=p["runs"]
        out["grid_"+n]={m:R[m]["from_MAP"]["params"] for m in ("0PA","PN","simple")}
    t0,tP,tS=_K("results_imri_tails_0pa.json"),_K("results_imri_tails_PN.json"),_K("results_imri_tails_simple.json")
    for n in t0:
        e={"0PA":t0[n]["params"]}
        for m,src in (("PN",tP),("simple",tS)):
            o=src[n]; e[m]=max((o["from_0PA"],o["from_NMdev"]),key=lambda r:r["ov_final"])["params"]
        out[n]=e
    aP,aS=_K("results_imri_PN.json"),_K("results_imri_simple.json")
    if "idx0" in aP:
        e={"0PA":[1.00176199e+06,4.99511288e+03,7.03962258e-01,2.49674581e+01,2.49798441e-01,
                  7.86137900e-01,1.00491369e+00,9.89953388e-01,6.42950551e-01]}
        for m,src in (("PN",aP),("simple",aS)):
            o=src["idx0"]; e[m]=max((o["from_0PA"],o["from_NMdev"]),key=lambda r:r["ov_final"])["params"]
        out["adhoc_A"]=e
    return out

# ---------------- cases ----------------
def build_cases():
    cs=[]; sig=np.load(SIGNAL_ARRAY)
    for i in range(25):
        r=sig[i]
        cs.append(dict(name=f"grid_idx{i}",group="grid",T=1.0,chi2=0.95,
            sp={"m1":1e6,"m2":1e3,"a":float(r[2]),"p0":float(r[3]),"e0":float(r[4]),"xI0":1.0,
                "dist":float(r[6]),"qS":float(r[7]),"phiS":float(r[8]),"qK":float(r[9]),
                "phiK":float(r[10]),"Phi_phi0":float(r[11]),"Phi_theta0":float(r[12]),"Phi_r0":float(r[13])}))
    b=dict(qS=1.04719755,phiS=0.78539816,qK=0.62831853,phiK=0.52359878,
           Phi_phi0=0.1,Phi_theta0=0.2,Phi_r0=0.3,xI0=1.0)
    cs.append(dict(name="pt4",group="tails",T=0.25,chi2=0.95,
        sp={**b,"m1":1e6,"m2":1e4,"a":0.9,"p0":29.2602456,"e0":0.10,"dist":272.11852}))
    cs.append(dict(name="pt20",group="tails",T=0.25,chi2=0.95,
        sp={**b,"m1":1e6,"m2":1e4,"a":-0.9,"p0":30.522071,"e0":0.50,"dist":94.592671}))
    cs.append(dict(name="adhoc_A",group="adhoc",T=1.0,chi2=0.95,
        sp={"m1":1e6,"m2":5.0e3,"a":0.70,"p0":25.0,"e0":0.25,"xI0":1.0,"dist":12.0,
            "qS":0.7853981633974483,"phiS":1.0,"qK":1.0,"phiK":1.0471975511965976,
            "Phi_phi0":0.9,"Phi_theta0":0.5,"Phi_r0":0.4}))
    return cs

def evaluate(case, fits, dt):
    sp,T,chi2 = case["sp"],case["T"],case["chi2"]
    rkw=dict(Tobs=T,t0=10000.0,dt=dt,index_lambda=8,index_beta=7,flip_hx=True,
             is_ecliptic_latitude=False,remove_garbage="zero",
             orbits=EqualArmlengthOrbits(use_gpu=USE_GPU),
             force_backend="cuda12x" if USE_GPU else "cpu",
             order=20,tdi="1st generation",tdi_chan="AET")
    wfm=GenerateEMRIWaveform(SuperKludgeWaveform,sum_kwargs=dict(pad_output=True,odd_len=True),
                             return_list=False,use_gpu=USE_GPU)
    wr=ResponseWrapper(waveform_gen=wfm,**rkw)
    def args(vec,names,e1pa,dev_on,tail=None):
        p={n:sp[n] for n in P14}; p.update(dict(zip(names,vec)))
        t=[chi2,e1pa,False,False,dev_on]
        if dev_on: t=t+tail(vec[9],vec[10])
        return [p[n] for n in P14]+t
    s=hp(xp.array(wr(*args([sp[n] for n in P9],P9,True,False)))[:NCH,:],dt)
    nk=[{"sens_fn":c} for c in CHAN[:NCH]]
    PSD=xp.array(generate_PSD(waveform=s,dt=dt,noise_PSD=get_sensitivity,channels=CHAN[:NCH],
                              noise_kwargs=nk,use_gpu=USE_GPU))
    fm=fmask(s.shape[-1],dt)
    ip=lambda a,bb,P: _f(inner_product(a,bb,PSD=P,dt=dt,freq_mask=fm,use_gpu=USE_GPU))
    per=[float(np.sqrt(max(ip(s[i:i+1],s[i:i+1],PSD[i:i+1]),0.0))) for i in range(NCH)]
    ss=ip(s,s,PSD); tot=float(np.sqrt(ss))
    res={}
    for m,vec in (fits or {}).items():
        try:
            if m=="0PA": h=xp.array(wr(*args(vec,P9,False,False)))[:NCH,:]
            else:        h=xp.array(wr(*args(vec,P11,False,True,TAIL[m])))[:NCH,:]
            r=s-h
            res[m]=dict(overlap=ip(s,h,PSD)/np.sqrt(ss*ip(h,h,PSD)), chi2=ip(r,r,PSD))
        except Exception as e:
            res[m]=dict(error=f"{type(e).__name__}: {e}")
    return per,tot,res

JSON_PATH=os.path.join(HERE,"snr_overlap_dt5_imri.json")

def main():
    fits=best_fits()
    print(f"[info] best-fit points loaded for {len(fits)} cases; re-evaluating at dt={DT_NEW}\n")
    out={"dt":DT_NEW,"note":"per-channel SNR and overlap/chi2 of dt=10-optimised best fits, re-evaluated at dt=5",
         "channels":NAMES,"generated_utc":datetime.now(timezone.utc).isoformat(),"cases":[]}
    print(f"{'case':12} {'a':>5} {'e0':>4} | {'SNR_A':>8} {'SNR_E':>8} {'SNR_T':>8} {'SNR_tot':>9} {'T%':>6} |"
          f" {'ov_0PA':>10} {'ov_PN':>10} {'ov_simple':>10}")
    print("-"*116)
    for c in build_cases():
        try:
            per,tot,res=evaluate(c,fits.get(c["name"]),DT_NEW)
        except Exception as e:
            print(f"{c['name']:12} FAILED {type(e).__name__}: {e}"); 
            out["cases"].append(dict(name=c["name"],error=str(e))); continue
        tp=100.0*(per[2]/tot)**2 if tot>0 else float("nan")
        g=lambda m: f"{res[m]['overlap']:10.7f}" if m in res and "overlap" in res[m] else f"{'--':>10}"
        print(f"{c['name']:12} {c['sp']['a']:+5.1f} {c['sp']['e0']:4.2f} | {per[0]:8.3f} {per[1]:8.3f} "
              f"{per[2]:8.3f} {tot:9.3f} {tp:6.2f} | {g('0PA')} {g('PN')} {g('simple')}")
        out["cases"].append(dict(name=c["name"],group=c["group"],a=c["sp"]["a"],e0=c["sp"]["e0"],
            dt=DT_NEW,T=c["T"],snr_A=per[0],snr_E=per[1],snr_T=per[2],snr_total=tot,
            T_power_percent=tp,models=res))
        with open(JSON_PATH,"w") as f: json.dump(out,f,indent=2)
    print("\n=== chi2 of the same best-fit points at dt=5 ===")
    print(f"{'case':12} {'chi2_0PA':>12} {'chi2_PN':>12} {'chi2_simple':>12}")
    for cc in out["cases"]:
        if "models" not in cc: continue
        g=lambda m: f"{cc['models'][m]['chi2']:12.4e}" if m in cc["models"] and "chi2" in cc["models"][m] else f"{'--':>12}"
        print(f"{cc['name']:12} {g('0PA')} {g('PN')} {g('simple')}")
    print(f"\n[saved] {JSON_PATH}")

if __name__=="__main__":
    main()
