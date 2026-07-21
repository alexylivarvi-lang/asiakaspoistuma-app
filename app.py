import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="AI Asiakaspoistuman Ennustin & ROI", page_icon="📊", layout="wide")

st.title("📊 AI Asiakaspoistuman Ennustin, Massatarkistus & ROI-laskuri")
st.write("Ennusta asiakaspoistumaa, generoi automaattiset toimenpiteet ja laske pelastettavissa oleva liikevaihto (ROI).")

# 1. MALLIN OPETUS JA DATA
@st.cache_resource
def train_model_and_get_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Ika': np.random.randint(18, 70, size=n_samples),
        'Kuukausimaksu': np.random.uniform(19.9, 119.9, size=n_samples).round(2),
        'Sopimustyyppi': np.random.choice(['Toistaiseksi', 'Maaraaikainen_12kk', 'Maaraaikainen_24kk'], size=n_samples),
        'Asiakaspalvelu_puhelut': np.random.randint(0, 8, size=n_samples)
    }
    df = pd.DataFrame(data)
    churn_prob = (0.1 + (df['Asiakaspalvelu_puhelut'] > 3) * 0.4 + (df['Sopimustyyppi'] == 'Toistaiseksi') * 0.25)
    df['Poistunut'] = (np.random.rand(n_samples) < churn_prob).astype(int)

    X = df.drop('Poistunut', axis=1)
    y = df['Poistunut']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Ika', 'Kuukausimaksu', 'Asiakaspalvelu_puhelut']),
            ('cat', OneHotEncoder(), ['Sopimustyyppi'])
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])
    pipeline.fit(X, y)
    
    result = permutation_importance(pipeline, X, y, n_repeats=5, random_state=42)
    importance_df = pd.DataFrame({
        'Muuttuja': X.columns,
        'Tarkeys': result.importances_mean
    }).sort_values(by='Tarkeys', ascending=True)

    return pipeline, importance_df

pipeline, importance_df = train_model_and_get_data()

# Sähköpostiluonnoksen generaattori
def luo_sahkopostiluonnos(ika, kuukausimaksu, sopimustyyppi, puhelut, poistumis_prob):
    if puhelut >= 4:
        syy = "Olet ollut useasti yhteydessä asiakaspalveluumme lähiaikoina."
        tarjous = "Haluamme varmistaa, että palvelumme toimii moitteettomasti ja tarjota sinulle maksuttoman palvelutarkistuksen sekä -20% alennuksen seuraavasta 3 kuukaudesta."
    elif sopimustyyppi == 'Toistaiseksi' and kuukausimaksu > 60:
        syy = "Arvostamme pitkäaikaista asiakkuuttasi."
        tarjous = "Haluamme tarjota sinulle mahdollisuuden päivittää sopimuksesi 12kk määräaikaissopimukseen erikoishintaan -25% nykyisestä kuukausimaksustasi."
    else:
        syy = "Olet meille erittäin tärkeä asiakas."
        tarjous = "Haluamme tarjota sinulle edun: saat seuraavan kuukauden veloituksetta reilun asiakkuutesi kunniaksi!"

    viesti = f"""
Hei!

{syy} {tarjous}

Ota etu käyttöön vastaamalla tähän viestiin tai klikkaamalla alla olevaa linkkiä:
[ Hyödynnä etu tästä ]

Ystävällisin terveisin,
Asiakaspuolen tiimi
"""
    return viesti

# Toimenpidesuosituksen määritys massadatalla
def maarita_toimenpide(row, kynnys):
    prob = row['Poistumisriski_raw']
    if prob < kynnys:
        return "✅ Ei toimenpiteitä"
    
    puhelut = row['Asiakaspalvelu_puhelut']
    maksu = row['Kuukausimaksu']
    
    if puhelut >= 4 and maksu > 70:
        return "🔥 VIP-asiakaspalvelun soitto & -25% etu"
    elif puhelut >= 4:
        return "📞 Asiakaspalvelun soittopyyntö (Tukipyyntö)"
    elif row['Sopimustyyppi'] == 'Toistaiseksi':
        return "✉️ Automaattinen -20% määräaikaispäivitystarjous"
    else:
        return "🎁 Sitouttamislahjakortti / Bonusetu"

# SIVUPALKKI: Hälytyskynnys & ROI-laskurin parametrit
st.sidebar.header("⚙️ Asetukset & Kynnys")
kynnys = st.sidebar.slider("Hälytyskynnys (Kynnysarvo)", 0.10, 0.90, 0.35, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("💰 ROI-Laskurin Oletukset")
onnistumisprosentti = st.sidebar.slider("Kampanjan onnistumis-%", 5, 50, 20, step=5) / 100.0
kampanja_kustannus_per_asiakas = st.sidebar.number_input("Kampanjan hinta / asiakas (€)", min_value=0.0, max_value=200.0, value=15.0, step=5.0)

# VÄLILEHDET
tab1, tab2 = st.tabs(["👤 Yksittäinen asiakas & Viestit", "📁 Massatarkistus & ROI"])

# --- VÄLILEHTI 1: YKSITTÄINEN ASIAKAS ---
with tab1:
    col_input, col_results = st.columns([1, 2])
    
    with col_input:
        st.subheader("Asiakkaan tiedot")
        ika = st.slider("Ikä", 18, 90, 35)
        kuukausimaksu = st.number_input("Kuukausimaksu (€)", min_value=10.0, max_value=200.0, value=79.90, step=5.0)
        sopimustyyppi = st.selectbox("Sopimustyyppi", ['Toistaiseksi', 'Maaraaikainen_12kk', 'Maaraaikainen_24kk'])
        puhelut = st.slider("Asiakaspalvelupuhelut (viim. 6kk)", 0, 10, 5)

        uusi_asiakas = pd.DataFrame({
            'Ika': [ika],
            'Kuukausimaksu': [kuukausimaksu],
            'Sopimustyyppi': [sopimustyyppi],
            'Asiakaspalvelu_puhelut': [puhelut]
        })
        poistumis_prob = pipeline.predict_proba(uusi_asiakas)[0][1]

    with col_results:
        st.subheader("📈 Ennusteen Yhteenveto")
        m1, m2 = st.columns(2)
        m1.metric(label="Poistumisriski", value=f"{poistumis_prob * 100:.1f} %")
        m2.metric(label="Valittu kynnys", value=f"{kynnys * 100:.0f} %")

        st.progress(float(poistumis_prob))

        if poistumis_prob >= kynnys:
            st.error("⚠️ **KORKEA POISTUMISRISKI!**")
            st.write("**Suositus:** Lähetä automaattinen sitouttamistarjous.")
        else:
            st.success("✅ **ASIAKUUS TURVASSA.**")
            st.write("**Suositus:** Ei tarvetta välittömille toimenpiteille.")

        fig1, ax1 = plt.subplots(figsize=(6, 1.8))
        ax1.barh(['Poistumisriski'], [poistumis_prob * 100], color='crimson' if poistumis_prob >= kynnys else 'mediumseagreen')
        ax1.axvline(kynnys * 100, color='black', linestyle='--', linewidth=2, label=f'Kynnys ({int(kynnys*100)}%)')
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('Prosenttia (%)')
        ax1.legend(loc='lower right')
        st.pyplot(fig1)

        # Generoitu sähköpostiluonnos
        if poistumis_prob >= kynnys:
            st.markdown("---")
            st.subheader("🤖 Tekoälyn generoima sähköpostiluonnos asiakkaalle")
            luonnos = luo_sahkopostiluonnos(ika, kuukausimaksu, sopimustyyppi, puhelut, poistumis_prob)
            st.code(luonnos, language="text")

# --- VÄLILEHTI 2: MASSATARKISTUS & ROI ---
with tab2:
    st.subheader("📁 Lataa asiakasrekisteri (CSV)")
    st.write("Lataa CSV-tiedosto, joka sisältää sarakkeet: `Ika`, `Kuukausimaksu`, `Sopimustyyppi`, `Asiakaspalvelu_puhelut`.")

    ladattu_tiedosto = st.file_uploader("Valitse CSV-tiedosto", type=["csv"])

    if ladattu_tiedosto is not None:
        try:
            df_massa = pd.read_csv(ladattu_tiedosto)
            tarvittavat_sarakkeet = ['Ika', 'Kuukausimaksu', 'Sopimustyyppi', 'Asiakaspalvelu_puhelut']
            
            puuttuvat = [col for col in tarvittavat_sarakkeet if col not in df_massa.columns]
            if puuttuvat:
                st.error(f"Tiedostosta puuttuvat seuraavat tarvittavat sarakkeet: {puuttuvat}")
            else:
                X_massa = df_massa[tarvittavat_sarakkeet]

                ennusteet_prob = pipeline.predict_proba(X_massa)[0:, 1]
                df_massa['Poistumisriski_raw'] = ennusteet_prob
                df_massa['Poistumisriski (%)'] = (ennusteet_prob * 100).round(1)
                df_massa['Hälytys'] = np.where(ennusteet_prob >= kynnys, '⚠️ KORKEA RISKI', '✅ Normaali')
                
                # Generoidaan automaattiset toimenpidesuositukset
                df_massa['Toimenpidesuositus'] = df_massa.apply(lambda row: maarita_toimenpide(row, kynnys), axis=1)

                st.markdown("---")
                st.subheader("📊 Massatarkistuksen Tulokset")

                riski_df = df_massa[df_massa['Poistumisriski_raw'] >= kynnys]
                korkea_riski_cnt = len(riski_df)
                yhteensa = len(df_massa)

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Asiakkaita yhteensä", yhteensa)
                col_m2.metric("Riskilistalla (yli kynnyksen)", korkea_riski_cnt)
                col_m3.metric("Riskiprosentti kannasta", f"{(korkea_riski_cnt / yhteensa) * 100:.1f} %")

                # ROI-LASKELMAT
                st.markdown("---")
                st.subheader("💰 Liiketoiminta- & ROI-laskelma")

                uhattu_mrr = riski_df['Kuukausimaksu'].sum()
                pelastettu_mrr = uhattu_mrr * onnistumisprosentti
                pelastettu_arr = pelastettu_mrr * 12
                
                # Kertaluonteinen kampanjakustannus
                kampanjan_kokonaiskustannus = korkea_riski_cnt * kampanja_kustannus_per_asiakas
                netto_saasto_vuosi = pelastettu_arr - kampanjan_kokonaiskustannus

                roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
                roi_col1.metric("Uhattu liikevaihto", f"{uhattu_mrr:,.2f} €/kk".replace(",", " "))
                roi_col2.metric(f"Pelastettu liikevaihto ({int(onnistumisprosentti*100)}%)", f"{pelastettu_mrr:,.2f} €/kk".replace(",", " "))
                roi_col3.metric("Vuositason säästö (ARR)", f"{pelastettu_arr:,.2f} €/v".replace(",", " "))
                roi_col4.metric("Nettosäästö kampanjan jälkeen", f"{netto_saasto_vuosi:,.2f} €/v".replace(",", " "), delta=f"{netto_saasto_vuosi:,.0f} €")

                st.info(f"💡 **Tulkinta:** Jos oletetaan, että kampanja kykenee sitouttamaan **{int(onnistumisprosentti*100)} %** riskiasiakkaista ja kertaluonteisen kampanjatoimenpiteen hinta on **{kampanja_kustannus_per_asiakas:.2f} €/asiakas**, investointi tuottaa **{netto_saasto_vuosi:,.2f} €** puhdasta vuosisäästöä.")

                # Esitetään siisti taulukko
                naytettava_df = df_massa.drop(columns=['Poistumisriski_raw']).sort_values(by='Poistumisriski (%)', ascending=False)

                st.write("### 🚨 Riskiasiakkaat ja Toimenpiteet:")
                st.dataframe(naytettava_df)

                # Latausnappi
                csv_data = naytettava_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Lataa valmis toimenpideraportti (CSV)",
                    data=csv_data,
                    file_name="asiakaspoistuma_toimenpiteet.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Virhe tiedoston käsittelyssä: {e}")
    else:
        st.info("💡 Voit ladata `asiakasdata.csv`-tiedoston kokeillaksesi ROI-laskuria!")
