############################################################
# CASPER V5 — Synthetic Recon Governance Console
# Deterministic • Human-Gated • Audit-Bound
#
# Recon-only synthetic ISR + corridor governance demo.
# No weapon control. No fire authority. No kinetic logic.
############################################################

import math
import time
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, replace
from typing import Dict, Any, List, Optional, Literal

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

DT_SECONDS = 1.0
MAX_HISTORY = 600

# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class AOConfig:
    label: str
    base_lat: float
    base_lon: float
    lat_delta: float
    lon_delta: float

@dataclass(frozen=True)
class EnvProfile:
    name: str
    latency_base: float
    latency_jitter: float
    thermal_bias: float
    imu_drift_bias: float

@dataclass(frozen=True)
class EnvelopePreset:
    name: str
    max_mach: float
    max_q_kpa: float
    max_g: float
    max_thermal_index: float
    max_latency_ms: float
    description: str

@dataclass(frozen=True)
class ThresholdPreset:
    name: str
    clarity_threshold: float
    threat_threshold: float
    description: str

@dataclass(frozen=True)
class MissionStage:
    code: str
    label: str
    duration: int
    description: str

@dataclass(frozen=True)
class Telemetry:
    tick: int
    utc_timestamp: str
    mission_time_s: float
    mission_stage_code: str
    mission_stage_label: str
    mission_stage_tick: int
    flight_phase: str

    mach: float
    velocity_mps: float
    altitude_m: float
    q_kpa: float
    thermal_index: float
    g_load: float
    link_latency_ms: float
    imu_drift_deg_s: float

    lat: float
    lon: float
    threat_index: float
    civ_density: float
    nav_drift: float
    comms_loss: float
    vision_hot_ratio: float

    clarity: float
    risk: float
    predicted_risk: float
    state: Literal["STABLE", "TENSE", "HIGH_RISK", "CRITICAL"]
    envelope_pressure: float

    cc_combined: float
    cc_nav_conf: float
    cc_comms_conf: float
    cc_vision_conf: float
    cc_clarity_factor: float
    cc_threat_factor: float

@dataclass
class EngineState:
    tick: int = 0
    mission_time_s: float = 0.0
    mission_stage_index: int = 0
    mission_stage_tick: int = 0

    run_id: int = 0
    rng_seed: int = 0

    ao: Optional[AOConfig] = None
    env_name: str = "Clear Skies / Clean Link"
    envelope_name: str = "Nominal Demo Flight"
    threshold_name: str = "Balanced"

    clarity_ema: float = 0.9
    history: List[Telemetry] = field(default_factory=list)
    terrain: Optional[np.ndarray] = None

    def utc(self):
        return datetime.utcnow().isoformat() + "Z"

# ============================================================
# PRESETS
# ============================================================

ENVELOPES = {
    "Nominal Demo Flight": EnvelopePreset("Nominal Demo Flight",1.8,650,4.5,0.78,300,"Balanced"),
    "Conservative Test Profile": EnvelopePreset("Conservative Test Profile",1.2,450,3.5,0.65,250,"Tight"),
    "Aggressive Envelope Probe": EnvelopePreset("Aggressive Envelope Probe",2.3,800,5.5,0.90,350,"Test only"),
}

ENVIRONMENTS = {
    "Clear Skies / Clean Link": EnvProfile("Clear",120,40,0.0,0.0),
    "High Latency Link": EnvProfile("High Lat",260,80,0.05,0.02),
}

THRESHOLDS = {
    "Balanced": ThresholdPreset("Balanced",75,65,"Standard"),
}

MISSION_STAGES = [
    MissionStage("STAGE_1_BOOST","Boost",40,""),
    MissionStage("STAGE_2_GRID","Grid",70,""),
    MissionStage("STAGE_3_RELAY","Relay",70,""),
    MissionStage("STAGE_4_COLLAPSE","Collapse",50,""),
    MissionStage("STAGE_5_RTB","RTB",9999,""),
]

AO_PRESETS = {
    "Kharkiv (synthetic)": AOConfig("Kharkiv Region",49.9935,36.2304,0.08,0.12),
}

# ============================================================
# HELPERS
# ============================================================

def clamp(v,a,b): return max(a,min(b,v))
def rng_for(state,t): return np.random.default_rng(state.rng_seed+t)

def ensure_terrain(state):
    if state.terrain is not None: return
    rng = np.random.RandomState(42)
    base = rng.normal(0,1,(80,80))
    for i in range(0,80,8):
        base[i:i+1]+=1.5; base[:,i:i+1]+=1.5
    state.terrain = base

# ============================================================
# CORE COMPUTATION
# ============================================================

def compute_clarity_and_risk(state,row):
    envp = ENVELOPES[state.envelope_name]
    q_u = clamp(row["q_kpa"]/envp.max_q_kpa,0,1.6)
    t_u = clamp(row["thermal_index"]/envp.max_thermal_index,0,1.8)
    threat_u = clamp(row["threat_index"]/100,0,1)

    pressure = 0.6*q_u + 0.4*t_u
    raw = clamp(1-pressure-0.3*threat_u,0.6,1.0)

    state.clarity_ema = 0.15*raw + 0.85*state.clarity_ema
    clarity = state.clarity_ema*100
    risk = clamp(pressure*60 + (100-clarity)*0.4,0,100)
    pred = clamp(risk + 8*(pressure-0.8),0,100)

    if clarity>=90 and risk<30: s="STABLE"
    elif clarity>=80: s="TENSE"
    elif clarity>=65: s="HIGH_RISK"
    else: s="CRITICAL"

    return clarity,risk,pred,s,round(pressure,3)

# ============================================================
# SYNTHETIC IR (RECON-ONLY)
# ============================================================

def render_ir(state,tel):
    ensure_terrain(state)
    base = state.terrain.copy()

    ao = state.ao
    x = int(40 + (tel.lon-ao.base_lon)*600)
    y = int(40 - (tel.lat-ao.base_lat)*600)
    x = clamp(x,2,77); y = clamp(y,2,77)

    base[y-1:y+2,x-1:x+2]+=4
    r = 6 + int(tel.threat_index*0.12)
    yy,xx = np.ogrid[:80,:80]
    base[(xx-x)**2+(yy-y)**2<=r*r]+=tel.threat_index/100*2

    vmin,vmax = np.percentile(base,5),np.percentile(base,95)
    img = np.clip((base-vmin)/(vmax-vmin),0,1)
    img *= (1-0.35*(1-tel.cc_combined))

    rgb = np.stack([img]*3,axis=-1)
    rgb = (rgb*255).astype(np.uint8)

    wm="SYNTHETIC — NOT OPERATIONAL"
    for i,_ in enumerate(wm):
        px=80-len(wm)*2+i*2; py=76
        if 0<=px<80: rgb[py:py+2,px:px+2]=200

    return rgb

# ============================================================
# STEP ENGINE
# ============================================================

def step(state):
    env = ENVIRONMENTS[state.env_name]
    rng = rng_for(state,state.tick+1)

    t = state.mission_time_s + DT_SECONDS
    mach = clamp((state.history[-1].mach if state.history else 0)+rng.uniform(0.01,0.05),0,2.3)
    alt = clamp((state.history[-1].altitude_m if state.history else 0)+rng.uniform(50,150),0,18000)

    vel = mach*295
    rho = 1.225*math.exp(-alt/8000)
    q = clamp(0.5*rho*vel**2/1000,0,900)

    thermal = clamp(0.2+0.5*(mach/2.3)+rng.normal(0,0.02),0,1)

    lat = state.ao.base_lat + rng.uniform(-state.ao.lat_delta,state.ao.lat_delta)
    lon = state.ao.base_lon + rng.uniform(-state.ao.lon_delta,state.ao.lon_delta)

    threat = clamp((state.history[-1].threat_index if state.history else 40)+rng.uniform(-5,5),0,100)
    civ = clamp((state.history[-1].civ_density if state.history else 0.3)+rng.uniform(-0.05,0.05),0,1)

    row={
        "mach":mach,"velocity_mps":vel,"altitude_m":alt,"q_kpa":q,
        "thermal_index":thermal,"g_load":1.0,
        "link_latency_ms":env.latency_base,"imu_drift_deg_s":0.02,
        "lat":lat,"lon":lon,"threat_index":threat,
        "civ_density":civ,"nav_drift":5,"comms_loss":0,
        "vision_hot_ratio":0.1
    }

    clarity,risk,pred,state_str,pressure = compute_clarity_and_risk(state,row)

    tel = Telemetry(
        tick=state.tick+1,
        utc_timestamp=state.utc(),
        mission_time_s=t,
        mission_stage_code="STAGE",
        mission_stage_label="Recon",
        mission_stage_tick=state.mission_stage_tick,
        flight_phase="CRUISE",
        mach=mach,velocity_mps=vel,altitude_m=alt,q_kpa=q,
        thermal_index=thermal,g_load=1.0,
        link_latency_ms=env.latency_base,imu_drift_deg_s=0.02,
        lat=lat,lon=lon,threat_index=threat,
        civ_density=civ,nav_drift=5,comms_loss=0,
        vision_hot_ratio=0.1,
        clarity=clarity,risk=risk,predicted_risk=pred,
        state=state_str,envelope_pressure=pressure,
        cc_combined=clarity/100,
        cc_nav_conf=0.9,cc_comms_conf=0.9,
        cc_vision_conf=0.8,cc_clarity_factor=clarity/100,
        cc_threat_factor=max(0.2,1-threat/150)
    )

    state.tick=tel.tick
    state.mission_time_s=t
    state.history.append(tel)
    state.history=state.history[-MAX_HISTORY:]
    return state

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="CASPER V5",layout="wide")

def get_state():
    if "state" not in st.session_state:
        s=EngineState()
        s.run_id=int(time.time()*1000)
        s.rng_seed=s.run_id%(2**32)
        s.ao=AO_PRESETS["Kharkiv (synthetic)"]
        st.session_state["state"]=s
    return st.session_state["state"]

state=get_state()

# ---- Header ----
c1,c2=st.columns([3,1])
with c1:
    st.markdown("## CASPER V5 — Synthetic Recon Governance Console")
    st.caption("Recon-only • deterministic • audit-bound")
with c2:
    st.markdown(
        f"<div style='font-family:monospace;font-size:12px;text-align:right;'>"
        f"RUN {state.run_id}<br/>SEED {state.rng_seed}</div>",
        unsafe_allow_html=True
    )

# ---- Controls ----
b1,b2=st.columns(2)
if b1.button("▶ Start"): st.session_state["run"]=True
if b2.button("⏸ Pause"): st.session_state["run"]=False

# ---- Step ----
if st.session_state.get("run",False):
    state=step(state)

if not state.history:
    st.info("Press Start to run CASPER V5.")
    st.stop()

tel=state.history[-1]

# ---- Metrics ----
m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Tick",tel.tick)
m2.metric("Clarity",f"{tel.clarity:.1f}%")
m3.metric("Risk (Now)",f"{tel.risk:.1f}%")
m4.metric("Risk (Pred)",f"{tel.predicted_risk:.1f}%")
m5.metric("State",tel.state)

st.markdown("---")

# ---- IR + Map ----
l,m,r=st.columns([1.2,1.0,1.2])
with l:
    st.subheader("Synthetic IR")
    st.image(render_ir(state,tel),use_container_width=True)
with r:
    st.subheader("Track Map")
    md=pd.DataFrame([[t.lat,t.lon] for t in state.history[-100:]],columns=["lat","lon"])
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer",data=md,get_position=["lon","lat"],get_radius=80)],
        initial_view_state=pdk.ViewState(latitude=state.ao.base_lat,longitude=state.ao.base_lon,zoom=10),
        map_style="mapbox://styles/mapbox/dark-v10"
    ))

# ---- Telemetry ----
st.markdown("---")
st.dataframe(pd.DataFrame([t.__dict__ for t in state.history[-40:]]))

if st.session_state.get("run",False):
    time.sleep(0.25)
    st.experimental_rerun()
