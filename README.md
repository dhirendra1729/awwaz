# AWWAZ (SAAKSHI / साक्षी)
### The Zero-Trust Voice Firewall for Financial Systems & Voice-Authorized Money Movement

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Zero--Trust_Voice_Firewall-green.svg)]()
[![Compliance](https://img.shields.io/badge/Compliance-DPDP_Act_2023_%7C_RBI_Cybersecurity-purple.svg)]()

---

## 0. Executive Summary & Core Thesis

Modern voice cloning tools (ElevenLabs, XTTS-v2, F5-TTS, CosyVoice) allow threat actors to clone voices from seconds of public audio to orchestrate high-stress **Authorized Push Payment (APP) Fraud**—such as kidnapping and accident extortion calls targeting families across India.

**The Core Thesis of SAAKSHI:**
> **Detection is probabilistic. Money is binary. Never let an acoustic detector alone move or block money.**

Security must derive from out-of-band cryptographic verification, contextual Bayesian prior-lifting, and reversible friction ladders.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SENSE (Probabilistic) ──► VERIFY (Deterministic) ──► ACT (Reversible) ──► PROVE (Lawful) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Key Innovations

1. **Contextual Bayesian Prior-Lifting**: Overcomes the base-rate paradox ($\pi = 10^{-5}$ yielding 1,250 false alarms per true attack) by gating on transactional and telephony state (`{Unknown Caller} ∩ {New Beneficiary VPA} ∩ {Amount > 3× Profile} ∩ {Call Active During UPI Session}`), elevating precision from $0.08\%$ to $>44.6\%$ without changing model weights.
2. **AWWAZ-AcousticNet**: Dual-branch architecture combining fast streaming edge SincNet filterbanks (<25ms CPU inference) with deep contextual embeddings, Temporal-Channel Modeling (TCM), and SpAArSIST graph attention.
3. **SafeEar Privacy Decoupler**: Disentangles speech embeddings into semantic linguistic tokens ($Z_{sem}$) and acoustic residuals ($Z_{ac}$). Discards $Z_{sem}$ every 500ms so no private banking credentials, PINs, or spoken conversations ever hit disk (DPDP Act 2023 compliant).
4. **Out-of-Band Family Life-Line Beacon (Method V2)**: Real-time silent cryptographic challenge to family members to vaporize psychological extortion in under 2 seconds.
5. **Emergency Rescue State Machine**: If a legitimate accident occurs, relatives can confirm the emergency to flip the hold to `ASSISTED_EMERGENCY_RELEASE`, prioritizing hospital payments and patching in senior bank concierges.
6. **Tamper-Evident SHA-256 Ledger & 1-Click I4C / 1930 Cybercrime Dossier**: Automates rapid reporting to the National Cybercrime Reporting Portal under golden-hour protocols with zero raw audio stored.

---

## 2. Repository Structure

```
awwaz/
├── README.md
├── pyproject.toml
├── configs/
│   ├── detectors.yaml               # Model ensemble weights and parameters
│   ├── channels.yaml                # Per-codec calibration parameters (G.711, AMR, Opus)
│   └── policy.yaml                  # Graduated action ladder rules & invariants
├── src/awwaz/
│   ├── audio/
│   │   ├── ingress.py               # Ring buffer & sliding window pipeline
│   │   ├── vad.py                   # Silero VAD ONNX wrapper
│   │   ├── codecs.py                # Telephony codec simulator (AMR, G.711, Opus)
│   │   └── privacy.py               # SafeEar semantic/acoustic decoupler
│   ├── models/
│   │   ├── sincnet.py               # 1D Learnable SincNet filterbank
│   │   ├── wavlm_backbone.py        # WavLM Large multi-layer feature extractor
│   │   ├── tcm.py                   # Temporal-Channel Modeling attention module
│   │   ├── spaarsist.py             # Sparsified Graph Attention pooling
│   │   └── acoustic_net.py          # Unified AWWAZ-AcousticNet architecture
│   ├── context/
│   │   ├── caller_intel.py          # CLI spoof likelihood & carrier lookup
│   │   ├── payment_profiler.py      # VPA novelty & amount anomaly detector
│   │   └── prior_lifter.py          # Bayesian contextual prior calculator
│   ├── verify/
│   │   ├── fido_push.py             # Android StrongBox / WebAuthn challenge mint
│   │   ├── family_beacon.py         # Out-of-band family lifeline protocol
│   │   └── tokens.py                # Ed25519 signed verification tokens
│   ├── policy/
│   │   ├── decision_engine.py       # Tri-zone decision matrix
│   │   └── action_ladder.py         # Level 0-6 reversible state machine
│   ├── prove/
│   │   ├── hash_chain.py            # SHA-256 tamper-evident ledger
│   │   └── i4c_dossier.py           # 1930 cybercrime reporting XML generator
│   └── api/
│       ├── main.py                  # FastAPI application entrypoint
│       └── websocket_stream.py      # Binary audio streaming endpoint
└── tests/
    ├── test_policy_no_single_signal.py  # Safety invariant: single signal cannot block
    ├── test_privacy_no_retention.py     # Privacy invariant: zero audio hits disk
    └── test_emergency_rescue.py         # Safety invariant: real emergencies release
```

---

## 3. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/dhirendra1729/awwaz.git
cd awwaz

# Setup python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest -v
```

### Launching the API Gateway

```bash
uvicorn awwaz.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 4. Architectural Deep-Dive

For full specifications and mathematical formulations:
- [AWWAZ_SAAKSHI_MASTER_BLUEPRINT.md](file:///home/dhiru/Documents/India/awwaz/AWWAZ_SAAKSHI_MASTER_BLUEPRINT.md): Master technical blueprint, scoring rubrics, 48h finale sprint plan.
- [RESEARCH_PAPERS_ANALYSIS_AND_TANA_CONTEXT.md](file:///home/dhiru/Documents/India/awwaz/RESEARCH_PAPERS_ANALYSIS_AND_TANA_CONTEXT.md): 20 research papers deep-dive with Tana supertags and benchmarking notes.
