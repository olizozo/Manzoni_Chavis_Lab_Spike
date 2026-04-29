import streamlit as st
import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tempfile
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Neural Excitability Pipeline", layout="wide")

# --- EN-TÊTE INSTITUTIONNEL (Harmonisé) ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: 
        st.image("logo_chavis_final.png", width=360) 
    except: 
        st.info("Manzoni Lab - Branding") 
with col_r:
    st.markdown("# Pipeline Expert : Excitabilité & Propriétés Intrinsèques")
    st.markdown("### Manzoni Lab | Analyse de la Plasticité Synaptique")
    st.markdown("#### *Extraction automatisée de la Rhéobase (I/V), Sag, Rin, Cm et Tau*")

st.divider()

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.header("📂 1. Chargement & Unités")
uploaded_file = st.sidebar.file_uploader("Charger un fichier ABF", type=["abf"])
current_unit = st.sidebar.radio("Unité du canal de courant (I_cmd)", ["pA", "nA"])

st.sidebar.header("⚙️ 2. Réglages de Détection")
spike_threshold = st.sidebar.number_input("Seuil de détection des PA (mV)", value=0.0)
dvdt_threshold = st.sidebar.number_input("Seuil dV/dt pour V_threshold (mV/ms)", value=15.0)

# --- LOGIQUE ANALYTIQUE COMPLÈTE ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        
        # 1. Extraction exhaustive des paramètres par sweep
        courants, voltages_stat, voltages_peak, voltages_rest = [], [], [], []
        spike_counts_raw, v_thresholds = [], []
        
        sr = abf.dataRate
        dt_ms = (1.0 / sr) * 1000.0  
        idx_start, idx_end = int(sr * 0.1), int(sr * 0.6)
        
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            i_cmd = np.mean(abf.sweepC[idx_start:idx_end])
            v_rest = np.mean(abf.sweepY[0:idx_start])
            v_stat = np.mean(abf.sweepY[idx_end - int(sr*0.05) : idx_end])
            
            # Peak : min pour hyperpolarisation, max pour dépolarisation
            v_peak = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
            
            # Détection des PA et Seuil dV/dt
            trace_window = abf.sweepY[idx_start:idx_end]
            peaks, _ = find_peaks(trace_window, height=spike_threshold)
            num_spikes = len(peaks)
            
            v_thresh_sweep = np.nan
            if num_spikes > 0:
                first_peak_idx = peaks[0]
                search_start = max(0, first_peak_idx - int(sr * 0.005))
                segment = trace_window[search_start:first_peak_idx]
                if len(segment) > 1:
                    dvdt = np.diff(segment) / dt_ms
                    crossings = np.where(dvdt > dvdt_threshold)[0]
                    v_thresh_sweep = segment[crossings[0]] if len(crossings) > 0 else trace_window[first_peak_idx]
            
            courants.append(i_cmd); voltages_stat.append(v_stat); voltages_peak.append(v_peak)
            voltages_rest.append(v_rest); spike_counts_raw.append(num_spikes); v_thresholds.append(v_thresh_sweep)

        # 2. Calculs Biophysiques Globaux (Rin, Tau, Cm, Rhéobase)
        v_rest_global = np.mean(voltages_rest)
        
        # Rhéobase
        rheobase_idx = next((i for i, count in enumerate(spike_counts_raw) if count > 0), None)
        rheobase_i = courants[rheobase_idx] if rheobase_idx is not None else None
        rheobase_v = v_thresholds[rheobase_idx] if rheobase_idx is not None else np.nan
        
        # Propriétés Passives
        neg_indices = [i for i, c in enumerate(courants) if c < 0]
        rin_mohm, tau_m_ms, cm_pf = np.nan, np.nan, np.nan
        if neg_indices:
            # Rin : Régression sur les pulses faibles
            neg_indices_sorted = sorted(neg_indices, key=lambda i: abs(courants[i]))[:4]
            i_neg = [courants[i] for i in neg_indices_sorted] + [0]
            v_neg = [voltages_stat[i] for i in neg_indices_sorted] + [v_rest_global]
            rin_mohm = np.polyfit(i_neg, v_neg, 1)[0] * (1 if current_unit == "nA" else 1000)
            
            # Tau : Calcul au 63% sur le pulse minimal
            idx_t = sorted(neg_indices, key=lambda i: abs(courants[i]))[0]
            abf.setSweep(idx_t)
            v_baseline = np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start])
            v_target = v_baseline + 0.632 * (voltages_stat[idx_t] - v_baseline)
            cross = np.where(abf.sweepY[idx_start:idx_end] <= v_target)[0]
            if len(cross) > 0:
                tau_m_ms = (cross[0] / sr) * 1000.0
                cm_pf = (tau_m_ms / rin_mohm) * 1000.0 if rin_mohm > 0 else np.nan

        # 3. TABLEAU DE BORD (Metrics)
        st.subheader("📊 Propriétés Intrinsèques Extraites")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vrest (Repos)", f"{v_rest_global:.1f} mV")
        c2.metric("Rin (Entrée)", f"{rin_mohm:.1f} MΩ" if not np.isnan(rin_mohm) else "N/A")
        c3.metric("Cm (Capacitance)", f"{cm_pf:.1f} pF" if not np.isnan(cm_pf) else "N/A")
        c4.metric("Tau_m (Constante)", f"{tau_m_ms:.1f} ms" if not np.isnan(tau_m_ms) else "N/A")
        
        c5, c6, c7 = st.columns(3)
        rheo_scientific = f"{rheobase_i * (1e-12 if current_unit == 'pA' else 1e-9):.2e} A" if rheobase_i else "N/A"
        c5.metric("Rhéobase (Intensité)", rheo_scientific)
        c6.metric("Rhéobase (Seuil mV)", f"{rheobase_v:.1f} mV" if not np.isnan(rheobase_v) else "N/A")
        c7.metric("Sag Amplitude (max)", f"{voltages_stat[np.argmin(courants)] - voltages_peak[np.argmin(courants)]:.1f} mV")

        st.divider()

        # 4. EXPORTATION DES DONNÉES (CSV multiples)
        st.subheader("📥 Exportation des Résultats")
        exp1, exp2 = st.columns(2)
        
        df_bio = pd.DataFrame({
            "Fichier": [uploaded_file.name], "Vrest_mV": [v_rest_global], "Rin_Mohm": [rin_mohm], 
            "Cm_pF": [cm_pf], "Tau_ms": [tau_m_ms], "Rheo_I_A": [rheo_scientific], 
            "Rheo_V_mV": [rheobase_v], "Sag_Max_mV": [voltages_stat[np.argmin(courants)] - voltages_peak[np.argmin(courants)]]
        })
        exp1.download_button("💾 Profil Biophysique Global (CSV)", df_bio.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_biophysique.csv", "text/csv", use_container_width=True)
        
        df_curv = pd.DataFrame({
            "Sweep": list(range(abf.sweepCount)), "I_inj": courants, "V_steady": voltages_stat, 
            "V_peak": voltages_peak, "V_threshold": v_thresholds, "Spikes_Raw": spike_counts_raw
        })
        exp2.download_button("📊 Données Courbes IV & f-I (CSV)", df_curv.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_donnees_courbes.csv", "text/csv", use_container_width=True)

        st.divider()

        # 5. VISUALISATION AVANCÉE (Individuelle & Overlay)
        st.subheader("📈 Exploration des Traces & Courbes")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            sw_idx = st.slider("Sélectionner un Sweep individuel", 0, abf.sweepCount - 1, 0)
        with col_v2:
            stk_indices = st.multiselect("Superposer des sweeps (Overlay)", list(range(abf.sweepCount)), default=[0, abf.sweepCount//2, abf.sweepCount-1])

        plt.switch_backend('Agg') 
        plt.style.use('seaborn-v0_8-paper')
        fig = plt.figure(figsize=(18, 14), dpi=110)
        gs = fig.add_gridspec(3, 2)
        
        def clean_ax(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=10)

        # Trace Individuelle
        ax0 = fig.add_subplot(gs[0, 0])
        abf.setSweep(sw_idx)
        ax0.plot(abf.sweepX, abf.sweepY, color='black', lw=1)
        ax0.set_title(f"Sweep {sw_idx} ({courants[sw_idx]:.1f} {current_unit})", fontweight='bold')
        if not np.isnan(v_thresholds[sw_idx]):
            ax0.axhline(v_thresholds[sw_idx], color='green', ls=':', label='Seuil')
        clean_ax(ax0)

        # Superposition (Overlay)
        ax1 = fig.add_subplot(gs[0, 1])
        cmap = plt.colormaps.get_cmap('viridis')
        for i, s in enumerate(stk_indices):
            abf.setSweep(s)
            ax1.plot(abf.sweepX, abf.sweepY, color=cmap(i/len(stk_indices)), lw=0.8, alpha=0.8)
        ax1.set_title(f"Overlay ({len(stk_indices)} sweeps)", fontweight='bold')
        clean_ax(ax1)
        
        # Courbe I-V
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(courants, voltages_stat, 'o-', label="Steady State")
        ax2.plot(courants, voltages_peak, '^--', alpha=0.4, label="Peak (Sag)")
        
        # RESTAURATION DU MARQUEUR VISUEL (Point rouge I-V)
        ax2.plot(courants[sw_idx], voltages_stat[sw_idx], 'ro', markersize=9, zorder=5) 
        
        ax2.axvline(0, color='gray', lw=0.5); ax2.axhline(v_rest_global, color='gray', lw=0.5, ls='--')
        ax2.set_title("Relation I-V (Passif & Sag)", fontweight='bold')
        ax2.set_xlabel(f"Injection ({current_unit})")
        ax2.set_ylabel("Vm (mV)")
        ax2.legend(frameon=False)
        clean_ax(ax2)
        
        # Courbe f-I
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(courants, spike_counts_raw, 's-', color='orange')
        
        # RESTAURATION DU MARQUEUR VISUEL (Point rouge f-I)
        ax3.plot(courants[sw_idx], spike_counts_raw[sw_idx], 'ro', markersize=9, zorder=5)
        
        if rheobase_i: ax3.axvline(rheobase_i, color='red', ls='--')
        ax3.set_title("Excitabilité : Courbe f-I", fontweight='bold')
        ax3.set_xlabel(f"Injection ({current_unit})")
        ax3.set_ylabel("Nombre de PA (Brut)")
        clean_ax(ax3)
        
        st.pyplot(fig)

        # 6. DOCUMENTATION INTÉGRÉE
        with st.expander("📖 Aide Mémoire : Formalisme & Biophysique"):
            st.markdown("""
            * **dV/dt Threshold :** Le seuil biophysique est défini comme l'instant où l'accélération du voltage (dérivée) dépasse la valeur cible (ex: 15 mV/ms), marquant l'ouverture massive des canaux Na+.
            * **Rin (MΩ) :** Calculé sur la pente de la portion linéaire hyperpolarisante (I < 0).
            * **Capacitance (pF) :** Déduite de la constante de temps $\\tau_m$ via la relation $C_m = \\tau_m / R_{in}$.
            * **Sag :** Différence entre le pic transitoire et l'état stationnaire, révélant l'activité des canaux HCN ($I_h$).
            """)

    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
else:
    st.info("Veuillez charger un fichier .abf pour activer le pipeline expert.")
