import streamlit as st
import pandas as pd
import anova_logic
import os
import json
import io

st.set_page_config(page_title="SPSS 가중 반복측정 ANOVA 분석 도구", layout="wide")

st.title("📊 SPSS 가중 반복측정 ANOVA 분석")

uploaded_file = st.file_uploader("SPSS (.sav) 파일을 업로드하세요", type=["sav"])

if uploaded_file is not None:
    # 임시 파일로 저장하여 pyreadstat에서 읽을 수 있게 함
    with open("temp.sav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    df, meta = anova_logic.load_spss_file("temp.sav")
    
    if df is not None:
        st.success(f"✅ 파일 로드 완료: {len(df)}개의 케이스")
        
        all_columns = list(df.columns)

        # --- 프리셋 관리 ---
        PRESET_FILE = "anova_presets.json"
        
        def load_presets():
            if os.path.exists(PRESET_FILE):
                try:
                    with open(PRESET_FILE, "r", encoding='utf-8') as f:
                        return json.load(f)
                except:
                    return {}
            return {}

        def save_preset(name, config):
            presets = load_presets()
            presets[name] = config
            with open(PRESET_FILE, "w", encoding='utf-8') as f:
                json.dump(presets, f, ensure_ascii=False, indent=4)

        st.sidebar.header("📂 분석 프리셋")
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
            st.sidebar.info(f"💡 '{selected_preset}' 프리셋이 로드되었습니다.")

        # --- 변수 선택 ---
        st.subheader("🔍 변수 설정")
        col1, col2 = st.columns(2)
        
        with col1:
            weight_col = st.selectbox("가중치(Weight) 변수 선택", all_columns, 
                                    index=all_columns.index(default_weight) if default_weight in all_columns else 0)
            
        with col2:
            dep_vars = st.multiselect("종속 변수 선택 (시점)", all_columns, default=default_deps,
                                    help="반복 측정된 여러 시점의 변수들을 선택하세요.")
            
        banner_vars = st.multiselect("배너(Banner) 변수 선택 (집단 구분)", all_columns, default=default_banners,
                                    help="결과를 나누어 보고 싶은 집단 변수들을 선택하세요.")

        # --- 프리셋 저장 UI ---
        with st.sidebar.expander("💾 현재 설정 저장"):
            new_preset_name = st.text_input("새 프리셋 이름")
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

        # --- 분석 옵션 ---
        st.subheader("⚙️ 분석 옵션")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            normalize = st.checkbox("가중치 정규화 (표본 크기 기준)", value=True, 
                                  help="가중치의 합이 실제 응답자 수(N)와 같아지도록 조정합니다. 통계적 검정력을 왜곡하지 않으려면 켜는 것이 권장됩니다.")
        with col_opt2:
            use_weighted_df = st.checkbox("가중치 기반 자유도 사용", value=True, 
                                        help="P-값 계산 시 실제 응답자 수 대신 가중치의 합(모집단 크기)을 자유도로 사용합니다. 데이터 복제 방식을 쓰지 않을 때 사용합니다.")
        with col_opt3:
            use_frequency_weight = st.checkbox("SPSS 방식 (데이터 복제)", value=True, 
                                             help="SPSS의 'Weight Cases'와 동일하게 가중치만큼 데이터를 물리적으로 복제하여 분석합니다. 가장 정확한 SPSS 재현 방식입니다.")

        if use_frequency_weight:
            st.info("ℹ️ **SPSS 방식(데이터 복제)**이 활성화되었습니다. 다른 옵션보다 우선하며, SPSS와 동일한 결과를 얻을 수 있습니다.")
        elif use_weighted_df and not normalize:
            st.warning("⚠️ '가중치 정규화'를 끄고 '가중치 기반 자유도'를 사용하면 모집단 크기에 의해 극단적인 P-값이 나올 수 있습니다.")

        if st.button("🚀 분석 실행"):
            if not weight_col or not dep_vars:
                st.error("가중치 변수와 최소 하나 이상의 종속 변수를 선택해야 합니다.")
            elif len(dep_vars) < 2:
                st.warning("반복측정 ANOVA를 위해 2개 이상의 종속 변수가 필요합니다.")
            else:
                results_data = []
                
                # 1. Total Sample
                res = anova_logic.weighted_repeated_measures_anova(df, dep_vars, weight_col, normalize=normalize, use_weighted_df=use_weighted_df, use_frequency_weight=use_frequency_weight)
                posthoc = anova_logic.calculate_posthoc_summary(df, dep_vars, weight_col, normalize=normalize, use_weighted_df=use_weighted_df, use_frequency_weight=use_frequency_weight)
                
                results_data.append({
                    "집단 (Group)": "전체 샘플 (Total)",
                    "가중 N": f"{res['weighted_n']:.2f}",
                    "F-값": f"{res['F']:.4f}",
                    "구형성 p": f"{res['m_p']:.4f}",
                    "p-값 (구형성가정)": f"{res['p_unc']:.4f}",
                    "p-값 (GG)": f"{res['p_gg']:.4f}",
                    "p-값 (HF)": f"{res['p_hf']:.4f}",
                    "사후검증 (본페로니)": posthoc
                })
                
                # 2. Banner Variables
                for banner in banner_vars:
                    val_labels = meta.variable_value_labels.get(banner, {}) if meta else {}
                    valid_df = df[df[banner].notna()]
                    groups = sorted(valid_df[banner].unique())
                    
                    for group_val in groups:
                        sub_df = df[df[banner] == group_val]
                        group_label = val_labels.get(group_val, str(group_val))
                        display_name = f"{banner}: {group_label}"
                        
                        res_sub = anova_logic.weighted_repeated_measures_anova(sub_df, dep_vars, weight_col, normalize=normalize, use_weighted_df=use_weighted_df, use_frequency_weight=use_frequency_weight)
                        posthoc_sub = anova_logic.calculate_posthoc_summary(sub_df, dep_vars, weight_col, normalize=normalize, use_weighted_df=use_weighted_df, use_frequency_weight=use_frequency_weight)
                        
                        results_data.append({
                            "집단 (Group)": display_name,
                            "가중 N": f"{res_sub['weighted_n']:.2f}",
                            "F-값": f"{res_sub['F']:.4f}",
                            "구형성 p": f"{res_sub['m_p']:.4f}",
                            "p-값 (구형성가정)": f"{res_sub['p_unc']:.4f}",
                            "p-값 (GG)": f"{res_sub['p_gg']:.4f}",
                            "p-값 (HF)": f"{res_sub['p_hf']:.4f}",
                            "사후검증 (본페로니)": posthoc_sub
                        })
                
                st.session_state['analysis_results'] = pd.DataFrame(results_data)

        # 결과 표시
        if 'analysis_results' in st.session_state:
            st.write("---")
            st.header("📋 분석 결과")
            results_df = st.session_state['analysis_results']
            st.dataframe(results_df, use_container_width=True)

            # XLSX 다운로드 기능
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

            st.subheader("💡 결과 해석 가이드")
            col_guide1, col_guide2 = st.columns(2)
            with col_guide1:
                st.markdown("""
                **1. 구형성 검정 (Mauchly's Test)**
                - **p > .05**: 구형성 가정을 충족합니다. `p-값 (구형성가정)`을 확인하세요.
                - **p < .05**: 구형성 가정이 위배되었습니다. 보정된 값을 사용하세요.
                """)
            with col_guide2:
                st.markdown("""
                **2. 보정값 선택 (Epsilon, ε)**
                - GG ε < 0.75 이면 `p-값 (GG)` 권장
                - GG ε > 0.75 이면 `p-값 (HF)` 권장
                """)
            
            st.info("⭐ **사후검증 표기 안내**: 숫자(1, 2, 3...)는 선택한 종속 변수의 순서를 의미합니다. 유의미한 차이(p < .05)가 있는 쌍만 표시됩니다.")
    
    # 임시 파일 삭제
    if os.path.exists("temp.sav"):
        try:
            os.remove("temp.sav")
        except:
            pass
