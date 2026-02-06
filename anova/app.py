import streamlit as st
import pandas as pd
import anova_logic
import os
import json
import io

st.set_page_config(page_title="SPSS Weighted Repeated Measures ANOVA", layout="wide")

st.title("SPSS Weighted Repeated Measures ANOVA")

uploaded_file = st.file_uploader("Upload .sav file", type=["sav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily to disk to be read by pyreadstat
    # pyreadstat reads from a path, so we need a temp file
    with open("temp.sav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    df, meta = anova_logic.load_spss_file("temp.sav")
    
    if df is not None:
        st.success(f"File loaded successfully: {len(df)} cases")
        
        all_columns = list(df.columns)

        # --- Preset Management ---
        PRESET_FILE = "anova_presets.json"
        
        def load_presets():
            if os.path.exists(PRESET_FILE):
                with open(PRESET_FILE, "r", encoding='utf-8') as f:
                    return json.load(f)
            return {}

        def save_preset(name, config):
            presets = load_presets()
            presets[name] = config
            with open(PRESET_FILE, "w", encoding='utf-8') as f:
                json.dump(presets, f, ensure_ascii=False, indent=4)

        st.sidebar.header("📊 분석 프리셋")
        presets = load_presets()
        preset_list = list(presets.keys())
        
        selected_preset = st.sidebar.selectbox("프리셋 선택", ["새로 만들기"] + preset_list)
        
        default_weight = all_columns[0] if all_columns else None
        default_deps = []
        default_banners = []
        
        if selected_preset != "새로 만들기":
            config = presets[selected_preset]
            default_weight = config.get("weight_col", default_weight)
            default_deps = [v for v in config.get("dep_vars", []) if v in all_columns]
            default_banners = [v for v in config.get("banner_vars", []) if v in all_columns]
            st.sidebar.info(f"'{selected_preset}' 프리셋이 로드되었습니다.")

        # --- Variable Selection ---
        col1, col2 = st.columns(2)
        
        with col1:
            weight_col = st.selectbox("Select Weight Variable", all_columns, index=all_columns.index(default_weight) if default_weight in all_columns else 0)
            
        with col2:
            dep_vars = st.multiselect("Select Dependent Variables (Time points)", all_columns, default=default_deps)
            
        banner_vars = st.multiselect("Select Banner Variables (Subgroups)", all_columns, default=default_banners)

        # --- Save Preset UI ---
        with st.sidebar.expander("💾 현재 설정 저장"):
            new_preset_name = st.text_input("프리셋 이름 입력")
            if st.button("프리셋 저장"):
                if new_preset_name:
                    config_to_save = {
                        "weight_col": weight_col,
                        "dep_vars": dep_vars,
                        "banner_vars": banner_vars
                    }
                    save_preset(new_preset_name, config_to_save)
                    st.success(f"'{new_preset_name}' 저장 완료!")
                    st.rerun()
                else:
                    st.error("이름을 입력하세요.")

        # --- Analysis Options ---
        st.subheader("Analysis Options")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            normalize = st.checkbox("가중치 정규화 (표본 크기 기준)", value=True, help="가중치의 합이 실제 응답자 수(N)와 같아지도록 조정합니다. 통계적 검정력을 왜곡하지 않으려면 켜는 것이 권장됩니다.")
        with col_opt2:
            use_weighted_df = st.checkbox("가중치 기반 자유도 사용", value=True, help="P-값 계산 시 실제 응답자 수 대신 가중치의 합(모집단 크기)을 자유도로 사용합니다. 데이터 복제 방식을 쓰지 않을 때 사용합니다.")
        with col_opt3:
            use_frequency_weight = st.checkbox("SPSS 방식 (데이터 복제)", value=True, help="SPSS의 'Weight Cases'와 동일하게 가중치만큼 데이터를 물리적으로 복제하여 분석합니다. 가장 정확한 SPSS 재현 방식입니다.")

        if use_frequency_weight:
            st.info("ℹ️ **SPSS 방식(데이터 복제)**이 활성화되었습니다. 다른 옵션보다 우선하며, SPSS와 동일한 N=1230 결과를 얻을 수 있습니다.")
        elif use_weighted_df and not normalize:
            st.warning("⚠️ '가중치 정규화'를 끄고 '가중치 기반 자유도'를 사용하면 모집단 크기에 의해 극단적인 P-값이 나올 수 있습니다.")

        if st.button("Run Analysis"):
            if not weight_col or not dep_vars:
                st.error("Please select a weight variable and at least one dependent variable.")
            else:
                results_data = []
                
                # 1. Total Sample
                res = anova_logic.weighted_repeated_measures_anova(df, dep_vars, weight_col, normalize=normalize, use_weighted_df=use_weighted_df, use_frequency_weight=use_frequency_weight)
                results_data.append({
                    "Group": "Total Sample",
                    "Weighted N": f"{res['weighted_n']:.2f}",
                    "F-Value": f"{res['F']:.4f}",
                    "Mauchly p": f"{res['m_p']:.4f}",
                    "P-Value(Assumed)": f"{res['p_unc']:.4f}",
                    "P-Value(GG)": f"{res['p_gg']:.4f}",
                    "P-Value(HF)": f"{res['p_hf']:.4f}",
                })
                
                # 2. Banner Variables
                for banner in banner_vars:
                    val_labels = meta.variable_value_labels.get(banner, {}) if meta else {}
                    valid_df = df[df[banner].notna()]
                    groups = sorted(valid_df[banner].unique())
                    
                    for group_val in groups:
                        sub_df = df[df[banner] == group_val]
                        group_label = val_labels.get(group_val, str(group_val))
                        display_name = f"{banner}={group_label}"
                        res_sub = anova_logic.weighted_repeated_measures_anova(sub_df, dep_vars, weight_col, normalize=normalize, use_weighted_df=use_weighted_df, use_frequency_weight=use_frequency_weight)
                        results_data.append({
                            "Group": display_name,
                            "Weighted N": f"{res_sub['weighted_n']:.2f}",
                            "F-Value": f"{res_sub['F']:.4f}",
                            "Mauchly p": f"{res_sub['m_p']:.4f}",
                            "P-Value(Assumed)": f"{res_sub['p_unc']:.4f}",
                            "P-Value(GG)": f"{res_sub['p_gg']:.4f}",
                            "P-Value(HF)": f"{res_sub['p_hf']:.4f}",
                        })
                
                st.session_state['analysis_results'] = pd.DataFrame(results_data)

        # 결과 표시 (세션 상태에 결과가 있으면 항상 표시)
        if 'analysis_results' in st.session_state:
            st.write("---")
            st.header("Analysis Results")
            results_df = st.session_state['analysis_results']
            st.table(results_df)

            # XLSX 다운로드 기능 (CSV에서 XLSX로 변경)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='ANOVA_Results')
            xlsx_data = output.getvalue()

            st.download_button(
                label="📥 분석 결과 다운로드 (Excel)",
                data=xlsx_data,
                file_name="anova_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            with st.expander("📊 결과 해석 가이드"):
                st.markdown("""
                - **Mauchly p > .05**: 구형성 가정을 충족합니다. **P-Value(Assumed)**를 보고하십시오.
                - **Mauchly p < .05**: 구형성 가정이 위배되었습니다. 보정된 값을 사용하십시오:
                    - Epsilon($\\epsilon$) < 0.75 이면 **P-Value(GG)** 권장
                    - Epsilon($\\epsilon$) > 0.75 이면 **P-Value(HF)** 권장
                """)
            
            st.info("💡 **SPSS 결과와 맞추는 팁**: SPSS의 'Weight By' 결과와 동일하게 만들려면 '가중치 정규화'를 끄고, '가중치 기반 자유도 사용'을 체크하십시오.")
    
    # Cleanup temp file
    if os.path.exists("temp.sav"):
        os.remove("temp.sav")
