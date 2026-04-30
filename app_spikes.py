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

# --- LANGUAGE SELECTION ---
lang = st.sidebar.selectbox("Language / Langue", ["Français", "English"])

# --- TRANSLATION DICTIONARY ---
T = {
    "Français": {
        "title": "Pipeline Expert : Excitabilité & Propriétés Intrinsèques",
        "subtitle": "Manzoni Lab | Traitement par Lot (Batch)",
        "desc": "*Extraction automatisée haut-débit de la Rhéobase, Sag, Rin, Cm et Tau*",
        "readme_link": "📖 [Documentation (README)](https://github.com/OliManzoni/Manzoni_Chavis_Lab_EPHYS_Stats/blob/main/README.md)",
        "doi_link": "🏷️ [DOI: 10.5281/zenodo.19912621](https://doi.org/10.5281/zenodo.19912621)",
        "sec1_load": "📂 1. Chargement (Batch)",
        "file_upload": "Charger un ou plusieurs fichiers ABF",
        "unit_radio": "Unité du canal de courant (I_cmd)",
        "sec2_settings": "⚙️ 2. Réglages de Détection",
        "spike_thresh": "Seuil de détection des PA (mV)",
        "dvdt_thresh": "Seuil dV/dt pour V_threshold (mV/ms)",
        "sec3_metrics": "📊 Propriétés Intrinsèques",
        "m_vrest": "Vrest",
        "m_rin": "Rin",
        "m_cm": "Cm",
        "m_tau": "Tau_m",
        "m_rheo_i": "Rhéobase (I)",
        "m_rheo_v": "Rhéobase (V)",
        "m_sag": "Sag Max",
        "sec4_export": "📥 Exportation Master (Lot Complet)",
        "btn_bio": "💾 Exporter le Profil Biophysique Global (CSV)",
        "btn_curv": "📊 Exporter les Courbes de toutes les cellules (CSV)",
        "sec5_viz": "📈 Exploration Visuelle",
        "viz_slider": "Sélectionner un Sweep individuel",
        "viz_multi": "Superposer des sweeps (Overlay)",
        "p_thresh": "Seuil",
        "p_iv_title": "Relation I-V",
        "p_iv_x": "Injection ({})",
        "p_fi_title": "Courbe f-I",
        "p_fi_y": "Nombre de PA",
        "help_title": "📖 Aide Mémoire : Formalisme & Biophysique",
        "help_text": """
            * **dV/dt Threshold :** Instant où l'accélération du voltage (dérivée) dépasse la valeur cible (marquant l'ouverture massive des canaux Na+).
            * **Rin (MΩ) :** Calculé sur la pente de la portion linéaire hyperpolarisante (I < 0).
            * **Capacitance (pF) :** Déduite de la constante de temps $\\tau_m$ via la relation $C_m = \\tau_m / R_{in}$.
            * **Sag :** Différence entre le pic transitoire et l'état stationnaire (activité des canaux HCN).
            """,
        "info_wait": "Veuillez charger vos fichiers .abf pour activer le pipeline de traitement."
    },
    "English": {
        "title": "Expert Pipeline: Excitability & Intrinsic Properties",
        "subtitle": "Manzoni Lab | Batch Processing",
        "desc": "*Automated high-throughput extraction of Rheobase, Sag, Rin, Cm, and Tau*",
        "readme_link": "📖 [Documentation (README)](https://github.com/OliManzoni/Manzoni_Chavis_Lab_EPHYS_Stats/blob/main/README.md)",
        "doi_link": "🏷️ [DOI: 10.5281/zenodo.19912621](https://doi.org/10.5281/zenodo.19912621)",
        "sec1_load": "📂 1. Batch Loading",
        "file_upload": "Upload one or multiple ABF files",
        "unit_radio": "Current channel unit (I_cmd)",
        "sec2_settings": "⚙️ 2. Detection Settings",
        "spike_thresh": "AP detection threshold (mV)",
        "dvdt_thresh": "dV/dt threshold for V_threshold (mV/ms)",
        "sec3_metrics": "📊 Intrinsic Properties",
        "m_vrest": "Vrest",
        "m_rin": "Rin",
        "m_cm": "Cm",
        "m_tau": "Tau_m",
        "m_rheo_i": "Rheobase (I)",
        "m_rheo_v": "Rheobase (V)",
        "m_sag": "Sag Max",
        "sec4_export": "📥 Master Export (Full Batch)",
        "btn_bio": "💾 Export Global Biophysical Profile (CSV)",
        "btn_curv": "📊 Export Curve Data for all cells (CSV)",
        "sec5_viz": "📈 Visual Exploration",
        "viz_slider": "Select individual Sweep",
        "viz_multi": "Overlay sweeps",
        "p_thresh": "Threshold",
        "p_iv_title": "I-V Relationship",
        "p_iv_x": "Injection ({})",
        "p_fi_title": "f-I Curve",
        "p_fi_y": "AP Count",
        "help_title": "📖 Cheat Sheet: Formalism & Biophysics",
        "help_text": """
            * **dV/dt Threshold:** The biophysical threshold is defined as the moment the voltage acceleration exceeds the target value.
            * **Rin (MΩ):** Calculated on the slope of the linear hyperpolarizing portion (I < 0).
            * **Capacitance (pF):** Deduced from the time constant $\\tau_m$ via the relationship $C_m = \\tau_m / R_{in}$.
            * **Sag:** Difference between the transient peak and steady state (HCN channel activity).
            """,
        "info_wait": "Please upload your .abf files to activate the processing pipeline."
    }
}[lang]


# --- EN-TÊTE INSTITUTIONNEL ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: 
        st.image("logo_chavis_final.png", width=360) 
    except: 
        st.info("Manzoni Lab - Branding") 
with col_r:
    st.markdown(f"# {T['title']}")
    st.markdown(f"### {T['subtitle']}")
    st.markdown(f"#### {T['desc']}")
    st.markdown(f"{T['readme_link']} &nbsp; | &nbsp; {T['doi_link']}")

st.divider()

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.header(T['sec1_load'])
# Modification Clé : accept_multiple_files=True
uploaded_files = st.sidebar.file_uploader(T['file_upload'], type=["abf"], accept_multiple_files=True)
current_unit = st.sidebar.radio(T['unit_radio'], ["pA", "nA"], index=1)

st.sidebar.header(T['sec2_settings'])
spike_threshold = st.sidebar.number_input(T['spike_thresh'], value=0.0)
dvdt_threshold = st.sidebar.number_input(T['dvdt_thresh'], value=15.0)

# --- STRUCTURE DE STOCKAGE GLOBAL ---
all_bio_data = []
all_curve_data = []

# --- LOGIQUE ANALYTIQUE BATCH ---
if uploaded_files:
    
    st.info(f"Traitement de {len(uploaded_files)} fichier(s) en cours...")
    
    for file_idx, uploaded_file in enumerate(uploaded_files):
        # Utilisation d'un expander pour empiler les analyses proprement
        with st.expander(f"Cellule : {uploaded_file.name}", expanded=(file_idx == 0)):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_filepath = tmp_file.name

            try:
                abf = pyabf.ABF(tmp_filepath)
                
                # 1. Extraction exhaustive des paramètres
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
                    
                    v_peak = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
                    
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

                # 2. Calculs Biophysiques Globaux
                v_rest_global = np.mean(voltages_rest)
                
                rheobase_idx = next((i for i, count in enumerate(spike_counts_raw) if count > 0), None)
                rheobase_i = courants[rheobase_idx] if rheobase_idx is not None else None
                rheobase_v = v_thresholds[rheobase_idx] if rheobase_idx is not None else np.nan
                
                neg_indices = [i for i, c in enumerate(courants) if c < 0]
                rin_mohm, tau_m_ms, cm_pf = np.nan, np.nan, np.nan
                if neg_indices:
                    neg_indices_sorted = sorted(neg_indices, key=lambda i: abs(courants[i]))[:4]
                    i_neg = [courants[i] for i in neg_indices_sorted] + [0]
                    v_neg = [voltages_stat[i] for i in neg_indices_sorted] + [v_rest_global]
                    rin_mohm = np.polyfit(i_neg, v_neg, 1)[0] * (1 if current_unit == "nA" else 1000)
                    
                    idx_t = sorted(neg_indices, key=lambda i: abs(courants[i]))[0]
                    abf.setSweep(idx_t)
                    v_baseline = np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start])
                    v_target = v_baseline + 0.632 * (voltages_stat[idx_t] - v_baseline)
                    cross = np.where(abf.sweepY[idx_start:idx_end] <= v_target)[0]
                    if len(cross) > 0:
                        tau_m_ms = (cross[0] / sr) * 1000.0
                        cm_pf = (tau_m_ms / rin_mohm) * 1000.0 if rin_mohm > 0 else np.nan

                sag_max = voltages_stat[np.argmin(courants)] - voltages_peak[np.argmin(courants)]
                rheo_scientific = rheobase_i * (1e-12 if current_unit == 'pA' else 1e-9) if rheobase_i else np.nan

                # Ajout aux Master DataFrames
                all_bio_data.append({
                    "Fichier": uploaded_file.name, "Vrest_mV": v_rest_global, "Rin_Mohm": rin_mohm, 
                    "Cm_pF": cm_pf, "Tau_ms": tau_m_ms, "Rheo_I_A": rheo_scientific, 
                    "Rheo_V_mV": rheobase_v, "Sag_Max_mV": sag_max
                })
                
                df_curv = pd.DataFrame({
                    "Fichier": [uploaded_file.name] * abf.sweepCount,
                    "Sweep": list(range(abf.sweepCount)), "I_inj": courants, "V_steady": voltages_stat, 
                    "V_peak": voltages_peak, "V_threshold": v_thresholds, "Spikes_Raw": spike_counts_raw
                })
                all_curve_data.append(df_curv)

                # 3. TABLEAU DE BORD (Metrics)
                st.markdown(f"**{T['sec3_metrics']}**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(T['m_vrest'], f"{v_rest_global:.1f} mV")
                c2.metric(T['m_rin'], f"{rin_mohm:.1f} MΩ" if not np.isnan(rin_mohm) else "N/A")
                c3.metric(T['m_cm'], f"{cm_pf:.1f} pF" if not np.isnan(cm_pf) else "N/A")
                c4.metric(T['m_tau'], f"{tau_m_ms:.1f} ms" if not np.isnan(tau_m_ms) else "N/A")
                
                c5, c6, c7 = st.columns(3)
                c5.metric(T['m_rheo_i'], f"{rheo_scientific:.2e} A" if not np.isnan(rheo_scientific) else "N/A")
                c6.metric(T['m_rheo_v'], f"{rheobase_v:.1f} mV" if not np.isnan(rheobase_v) else "N/A")
                c7.metric(T['m_sag'], f"{sag_max:.1f} mV")

                st.divider()

                # 4. VISUALISATION AVANCÉE
                st.markdown(f"**{T['sec5_viz']}**")
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    # Ajout de clés uniques basées sur le nom du fichier pour éviter les conflits Streamlit
                    sw_idx = st.slider(T['viz_slider'], 0, abf.sweepCount - 1, 0, key=f"slide_{uploaded_file.name}")
                with col_v2:
                    stk_indices = st.multiselect(T['viz_multi'], list(range(abf.sweepCount)), default=[0, abf.sweepCount//2, abf.sweepCount-1], key=f"multi_{uploaded_file.name}")

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
                    ax0.axhline(v_thresholds[sw_idx], color='green', ls=':', label=T['p_thresh'])
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
                ax2.plot(courants[sw_idx], voltages_stat[sw_idx], 'ro', markersize=9, zorder=5) 
                
                ax2.axvline(0, color='gray', lw=0.5); ax2.axhline(v_rest_global, color='gray', lw=0.5, ls='--')
                ax2.set_title(T['p_iv_title'], fontweight='bold')
                ax2.set_xlabel(T['p_iv_x'].format(current_unit))
                ax2.set_ylabel("Vm (mV)")
                ax2.legend(frameon=False)
                clean_ax(ax2)
                
                # Courbe f-I
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.plot(courants, spike_counts_raw, 's-', color='orange')
                ax3.plot(courants[sw_idx], spike_counts_raw[sw_idx], 'ro', markersize=9, zorder=5)
                
                if rheobase_i: ax3.axvline(rheobase_i, color='red', ls='--')
                ax3.set_title(T['p_fi_title'], fontweight='bold')
                ax3.set_xlabel(T['p_iv_x'].format(current_unit))
                ax3.set_ylabel(T['p_fi_y'])
                clean_ax(ax3)
                
                st.pyplot(fig)
                plt.close(fig) # Fermeture explicite de la figure pour la gestion mémoire du batch

            except Exception as e:
                st.error(f"Erreur lors de l'analyse du fichier {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(tmp_filepath): os.remove(tmp_filepath)

    # --- 5. EXPORTATION MASTER BATCH ---
    if all_bio_data:
        st.divider()
        st.header(T['sec4_export'])
        
        master_bio_df = pd.DataFrame(all_bio_data)
        master_curve_df = pd.concat(all_curve_data, ignore_index=True)
        
        exp1, exp2 = st.columns(2)
        exp1.download_button(
            T['btn_bio'], 
            master_bio_df.to_csv(index=False).encode('utf-8'), 
            "Lot_Global_Biophysique.csv", 
            "text/csv", 
            use_container_width=True
        )
        
        exp2.download_button(
            T['btn_curv'], 
            master_curve_df.to_csv(index=False).encode('utf-8'), 
            "Lot_Global_Courbes.csv", 
            "text/csv", 
            use_container_width=True
        )

    with st.sidebar.expander(T['help_title']):
        st.markdown(T['help_text'])

else:
    st.info(T['info_wait'])
