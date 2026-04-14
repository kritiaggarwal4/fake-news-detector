import streamlit as st
import numpy as np
import pandas as pd
import requests
import re
import os
import pickle
import zipfile
import io
from datetime import datetime

# ML imports
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# LIME
from lime.lime_text import LimeTextExplainer

# Visualization
import plotly.graph_objects as go

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# URL scraping
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED CONFIG  (no user input needed)
# ─────────────────────────────────────────────────────────────────────────────
NEWSAPI_KEY  = "ab079b92c42e4ce18d143df65097866a"
NEWSAPI_URL  = "https://newsapi.org/v2/everything"
NEWSAPI_HEADLINES = "https://newsapi.org/v2/top-headlines"
MODEL_CACHE  = "model_cache.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection App",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg:#0f1117; --surface:#1a1d27; --border:#2a2d3e;
  --accent:#4f8ef7; --accent2:#22d3a0; --danger:#ff4d6d;
  --warning:#fbbf24; --text:#e8eaf0; --muted:#8b8fa8;
}
html,body,.stApp{background:var(--bg)!important;font-family:'Sora',sans-serif!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:2rem!important;max-width:960px!important;margin:0 auto;}

/* HERO */
.hero{text-align:center;padding:3rem 1rem 2rem;
  background:linear-gradient(135deg,rgba(79,142,247,.08),rgba(34,211,160,.05));
  border:1px solid var(--border);border-radius:20px;margin-bottom:2rem;}
.hero-icon{font-size:3.5rem;margin-bottom:.5rem;}
.hero h1{font-size:2.6rem;font-weight:800;
  background:linear-gradient(135deg,#4f8ef7,#22d3a0);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin:0;letter-spacing:-1px;}
.hero p{color:var(--muted);font-size:1rem;margin-top:.5rem;}

/* RESULT BADGES */
.result-real{display:inline-block;background:linear-gradient(135deg,#064e3b,#065f46);
  border:2px solid #22d3a0;color:#22d3a0;font-size:2rem;font-weight:800;
  padding:.6rem 2rem;border-radius:12px;letter-spacing:1px;margin:1rem 0;}
.result-fake{display:inline-block;background:linear-gradient(135deg,#7f1d1d,#991b1b);
  border:2px solid #ff4d6d;color:#ff4d6d;font-size:2rem;font-weight:800;
  padding:.6rem 2rem;border-radius:12px;letter-spacing:1px;margin:1rem 0;}
.result-uncertain{display:inline-block;background:linear-gradient(135deg,#78350f,#92400e);
  border:2px solid #fbbf24;color:#fbbf24;font-size:2rem;font-weight:800;
  padding:.6rem 2rem;border-radius:12px;letter-spacing:1px;margin:1rem 0;}

/* BUTTONS */
.stButton>button{background:linear-gradient(135deg,#4f8ef7,#3b82f6)!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  padding:.6rem 2rem!important;font-family:'Sora',sans-serif!important;
  font-weight:600!important;font-size:1rem!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;
  box-shadow:0 8px 20px rgba(79,142,247,.4)!important;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;
  border-radius:10px;border:1px solid var(--border);gap:4px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;
  font-family:'Sora',sans-serif!important;font-weight:600!important;border-radius:8px!important;}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:#fff!important;}

/* INPUTS */
.stTextArea textarea,.stTextInput input{
  background:var(--surface)!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:12px!important;
  font-family:'Sora',sans-serif!important;}

/* METRICS */
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:1rem;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.8rem!important;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-weight:700!important;}

.summary-box{background:rgba(79,142,247,.06);border:1px solid rgba(79,142,247,.2);
  border-radius:12px;padding:1rem 1.2rem;font-size:.95rem;line-height:1.6;margin:.8rem 0;}
.news-match{background:rgba(34,211,160,.05);border:1px solid rgba(34,211,160,.2);
  border-radius:10px;padding:.8rem 1rem;margin:.5rem 0;font-size:.88rem;}
.news-no-match{background:rgba(255,77,109,.05);border:1px solid rgba(255,77,109,.2);
  border-radius:10px;padding:.8rem 1rem;margin:.5rem 0;font-size:.88rem;}
.lime-positive{background:rgba(34,211,160,.1);border-left:3px solid #22d3a0;
  padding:.4rem .8rem;margin:.3rem 0;border-radius:0 8px 8px 0;
  font-family:'JetBrains Mono',monospace;font-size:.85rem;}
.lime-negative{background:rgba(255,77,109,.1);border-left:3px solid #ff4d6d;
  padding:.4rem .8rem;margin:.3rem 0;border-radius:0 8px 8px 0;
  font-family:'JetBrains Mono',monospace;font-size:.85rem;}
.section-label{font-size:.75rem;font-weight:700;text-transform:uppercase;
  letter-spacing:2px;color:var(--muted);margin-bottom:.5rem;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    for pkg in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

setup_nltk()

# ─────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        sw = set(stopwords.words('english'))
        lemma = WordNetLemmatizer()
        tokens = text.split()
        tokens = [lemma.lemmatize(w) for w in tokens if w not in sw and len(w) > 2]
        text = ' '.join(tokens)
    except Exception:
        pass
    return text

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER  (LIAR  +  FakeNewsNet  +  fallback built-in)
# ─────────────────────────────────────────────────────────────────────────────
def load_liar_dataset() -> pd.DataFrame:
    """
    Loads LIAR dataset from local TSV files if available.
    Place train.tsv / valid.tsv / test.tsv inside  data/liar/
    Download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    """
    dfs = []
    base = os.path.join("data", "liar")
    for fname in ["train.tsv", "valid.tsv", "test.tsv"]:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            try:
                cols = [
                    "id","label","statement","subjects","speaker",
                    "job","state","party","barely_true","false",
                    "half_true","mostly_true","pants_fire","context"
                ]
                df = pd.read_csv(fpath, sep='\t', header=None, names=cols)
                # Map LIAR 6-class → binary  (real=1, fake=0)
                real_labels  = {'true','mostly-true','half-true'}
                fake_labels  = {'false','pants-fire','barely-true'}
                df = df[df['label'].isin(real_labels | fake_labels)].copy()
                df['binary'] = df['label'].apply(lambda x: 1 if x in real_labels else 0)
                df['text']   = df['statement'].astype(str)
                dfs.append(df[['text','binary']])
            except Exception as e:
                st.warning(f"⚠️ Could not load {fname}: {e}")

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        result.columns = ['text','label']
        return result
    return pd.DataFrame()

def load_fakenewsnet_dataset() -> pd.DataFrame:
    """
    Loads FakeNewsNet CSV files if available.
    Place politifact_fake.csv / politifact_real.csv /
          gossipcop_fake.csv  / gossipcop_real.csv
    inside  data/fakenewsnet/
    Download: https://github.com/KaiDMML/FakeNewsNet
    """
    dfs = []
    base = os.path.join("data", "fakenewsnet")
    for fname, label in [
        ("politifact_fake.csv",0), ("politifact_real.csv",1),
        ("gossipcop_fake.csv", 0), ("gossipcop_real.csv", 1),
    ]:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                # FakeNewsNet has 'title' column
                text_col = next((c for c in ['title','news_url','text'] if c in df.columns), None)
                if text_col:
                    df = df[[text_col]].dropna().copy()
                    df.columns = ['text']
                    df['label'] = label
                    dfs.append(df)
            except Exception as e:
                st.warning(f"⚠️ Could not load {fname}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def load_kaggle_dataset() -> pd.DataFrame:
    """
    Loads Kaggle Fake-and-Real News Dataset if available.
    Place Fake.csv and True.csv inside  data/kaggle/
    Download: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    """
    dfs = []
    base = os.path.join("data", "kaggle")
    for fname, label in [("Fake.csv", 0), ("True.csv", 1)]:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                text_cols = [c for c in ['title','text'] if c in df.columns]
                if text_cols:
                    df['combined'] = df[text_cols].fillna('').agg(' '.join, axis=1)
                    df = df[['combined']].copy()
                    df.columns = ['text']
                    df['label'] = label
                    dfs.append(df)
            except Exception as e:
                st.warning(f"⚠️ Could not load {fname}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def get_builtin_dataset() -> pd.DataFrame:
    """
    Large built-in dataset used when no local files are found.
    Covers diverse real and fake patterns for decent accuracy.
    """
    real = [
        "Scientists have confirmed a new vaccine showing 95 percent efficacy in phase three clinical trials published in the New England Journal of Medicine.",
        "The Federal Reserve raised interest rates by 25 basis points to address persistent inflation exceeding the two percent target.",
        "NASA successfully launched the Artemis mission carrying astronauts toward lunar orbit for the first time since Apollo 17.",
        "Parliamentary elections in Germany concluded with the Social Democrats winning a narrow majority according to official results.",
        "Researchers at MIT published peer-reviewed findings on improved solar panel efficiency reaching 47 percent conversion rates.",
        "The World Health Organization declared an end to the mpox public health emergency of international concern.",
        "A magnitude 6.8 earthquake struck coastal Japan triggering tsunami warnings that were later lifted by authorities.",
        "The Supreme Court ruled 6-3 that the administrative agency had exceeded its statutory authority in the landmark case.",
        "The unemployment rate fell to 3.4 percent in December, the lowest reading in over 50 years according to the Bureau of Labor Statistics.",
        "Apple reported quarterly earnings of 89 billion dollars, beating analyst estimates by 4 percent in its fiscal first quarter.",
        "Scientists detected gravitational waves from a neutron star merger using the LIGO and Virgo observatories.",
        "The International Monetary Fund revised global growth forecasts downward to 2.9 percent for the current fiscal year.",
        "Climate scientists recorded the hottest global average temperature in 125,000 years during last summer.",
        "SpaceX successfully launched the Falcon Heavy rocket carrying a classified military satellite into geostationary orbit.",
        "The European Central Bank announced it would maintain accommodative monetary policy through the third quarter.",
        "Public health officials confirmed measles vaccination rates have recovered to pre-pandemic levels in most countries.",
        "University researchers demonstrated a new antibiotic effective against drug-resistant bacteria in laboratory settings.",
        "The trade balance widened to its largest deficit since 2008 amid surging import demand and slower export growth.",
        "Local elections across three major cities resulted in incumbent mayors retaining their positions with comfortable margins.",
        "Engineers completed the first intercontinental quantum communication experiment using satellite relay technology.",
        "A new peer-reviewed study in the Lancet links ultraprocessed food consumption to elevated cardiovascular disease risk.",
        "The central bank governor testified before Congress regarding the outlook for monetary tightening over coming quarters.",
        "Flooding in coastal regions affected an estimated 200,000 residents according to disaster management officials.",
        "Pharmaceutical regulators approved the first oral treatment for sickle cell disease after successful clinical trials.",
        "Astronomers discovered an exoplanet in the habitable zone of its star with signs of water vapor in its atmosphere.",
        "The labor department reported wage growth of 4.1 percent year over year, outpacing inflation for the first time.",
        "A bipartisan infrastructure bill passed the Senate with 69 votes allocating 550 billion in new federal spending.",
        "The technology company announced layoffs affecting eight percent of its global workforce amid declining revenues.",
        "Health authorities reported a significant decline in antibiotic-resistant infections following new hospital protocols.",
        "The World Trade Organization ruled that retaliatory tariffs violated international trade agreements in the dispute.",
        "Census data revealed the fastest urban population growth in a decade concentrated in southern and southwestern cities.",
        "Scientists successfully edited a gene associated with inherited blindness in a clinical trial showing restored vision.",
        "The international climate agreement was ratified by 165 countries committing to net zero emissions by 2050.",
        "A new battery technology demonstrated energy density three times greater than current lithium-ion cells in tests.",
        "Global renewable energy capacity surpassed fossil fuels for the first time in total installed generation capacity.",
    ]

    fake = [
        "BOMBSHELL: Leaked documents prove that the COVID-19 vaccine contains microchips to track all citizens globally!!",
        "SHOCKING PROOF: The moon landing was staged by Stanley Kubrick in a secret Hollywood studio funded by NASA!!",
        "BREAKING: Doctors CONFIRM that drinking bleach mixed with lemon juice CURES cancer in just three days!!",
        "EXPOSED: 5G towers are secretly transmitting mind control frequencies to make people obedient to the New World Order!!",
        "ALERT: Bill Gates personally funded the development of a bioweapon disguised as the coronavirus pandemic!!",
        "REVEALED: The flat earth has been covered up by NASA, airlines, and governments for over 200 years!!",
        "MIRACLE CURE: This simple herb found in the Amazon dissolves tumors overnight and big pharma is hiding it!!",
        "URGENT WARNING: New government law will force all citizens to implant digital ID chips or lose their bank accounts!!",
        "CONFIRMED: Hollywood elites are running a secret underground child trafficking network out of a pizza restaurant!!",
        "PROOF: Chemtrails from commercial aircraft are spreading sterilization chemicals to reduce the global population!!",
        "INSIDER REVEALS: Soros funded thousands of crisis actors to fake shootings and push gun confiscation agenda!!",
        "SHOCKING: Fluoride in the water supply is a mind-numbing agent used to make the population docile and obedient!!",
        "BOMBSHELL REPORT: The government is using cell phone towers to broadcast silent cancer-causing radiation!!",
        "ALERT: Scientists admit vaccines cause autism but were paid billions to suppress the research from parents!!",
        "EXCLUSIVE: JFK Jr is alive and working with patriots underground to expose the deep state cabal worldwide!!",
        "BREAKING: Antarctica is actually a massive wall surrounding the flat earth that governments forbid you from visiting!!",
        "MIRACLE: Man regrew his lost arm using a combination of apple cider vinegar and turmeric paste in two weeks!!",
        "EXPOSED: The mainstream media is completely controlled by six globalist corporations pushing a single agenda!!",
        "SECRET LEAKED: Area 51 houses over 50 living alien species in partnership with reptilian government officials!!",
        "WARNING: The new digital currency will give central banks complete control over every purchase you ever make!!",
        "CONFIRMED: Famous singer was replaced by a government clone after refusing to push the globalist agenda in songs!!",
        "REVEALED: Drinking alkaline water cures every disease known to man and oncologists are hiding the truth from you!!",
        "BREAKING: A suppressed Tesla energy device can power your entire home for free and oil companies buried the patent!!",
        "ALERT: New study proves that sunscreen causes cancer while the UV protection story was invented by pharma companies!!",
        "EXCLUSIVE: The Titanic never sank, it was sunk deliberately as part of an insurance scam to eliminate opponents of the Federal Reserve!!",
        "SHOCK REVEAL: Doctors prescribe unnecessary surgery and chemotherapy to profit from your illness rather than cure you!!",
        "BOMBSHELL: Reptilian shapeshifters have infiltrated the highest levels of every major government on earth!!",
        "PROOF: Eating magnets aligns your body electromagnetic field and reverses aging according to suppressed research!!",
        "BREAKING: The Great Wall of China was built by an ancient race of giants and archaeologists are covering it up!!",
        "EXPOSED: Social media platforms are secretly recording your conversations 24/7 and selling them to foreign governments!!",
        "ALERT: New legislation secretly embedded in the infrastructure bill will allow drones to monitor all private citizens!!",
        "SHOCKING: Hospital death panels are euthanizing elderly patients to reduce healthcare costs under secret protocols!!",
        "CONFIRMED: The deep state used weather weapons to manufacture Hurricane Katrina as a population control experiment!!",
        "LEAKED: Major pharmaceutical company secretly paid celebrities to endorse vaccines they knew were dangerous!!",
        "WARNING: The real unemployment rate is 40 percent and the government is fabricating official statistics to avoid panic!!",
    ]

    texts  = real * 3 + fake * 3
    labels = [1] * (len(real)*3) + [0] * (len(fake)*3)
    return pd.DataFrame({'text': texts, 'label': labels})

def build_combined_dataset() -> pd.DataFrame:
    frames = []

    liar = load_liar_dataset()
    if not liar.empty:
        frames.append(liar)

    fnn = load_fakenewsnet_dataset()
    if not fnn.empty:
        frames.append(fnn)

    kag = load_kaggle_dataset()
    if not kag.empty:
        frames.append(kag)

    builtin = get_builtin_dataset()
    frames.append(builtin)

    df = pd.concat(frames, ignore_index=True).dropna(subset=['text','label'])
    df['label'] = df['label'].astype(int)
    df = df.drop_duplicates(subset=['text'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# HIGH-ACCURACY MODEL  (Voting Ensemble: LR + GBM + RF)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if os.path.exists(MODEL_CACHE):
        try:
            with open(MODEL_CACHE, 'rb') as f:
                cache = pickle.load(f)
            return cache['clf'], cache['tfidf'], cache['acc'], cache['dataset_size']
        except Exception:
            pass

    # ── Build dataset ──────────────────────────────────────────────────────
    df = build_combined_dataset()
    dataset_size = len(df)

    df['processed'] = df['text'].apply(preprocess)

    # ── TF-IDF (character + word n-grams for robustness) ──────────────────
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        analyzer='word',
        stop_words='english',
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )

    X = tfidf.fit_transform(df['processed'])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # ── Individual learners ────────────────────────────────────────────────
    lr = LogisticRegression(
        C=5.0, solver='lbfgs', max_iter=1000,
        class_weight='balanced', random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_split=2, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    # GBM only if dataset is small enough to train quickly
    if dataset_size < 20000:
        gbm = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=5, random_state=42
        )
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('gbm', gbm)],
            voting='soft', weights=[3, 2, 2]
        )
    else:
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf)],
            voting='soft', weights=[3, 2]
        )

    ensemble.fit(X_train, y_train)
    acc = accuracy_score(y_test, ensemble.predict(X_test))

    with open(MODEL_CACHE, 'wb') as f:
        pickle.dump({'clf': ensemble, 'tfidf': tfidf, 'acc': acc, 'dataset_size': dataset_size}, f)

    return ensemble, tfidf, acc, dataset_size

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION & LIME
# ─────────────────────────────────────────────────────────────────────────────
def predict_news(text, clf, tfidf):
    processed = preprocess(text)
    X = tfidf.transform([processed])
    proba = clf.predict_proba(X)[0]
    pred  = int(np.argmax(proba))
    confidence = float(max(proba)) * 100
    return pred, confidence, proba

def get_lime_explanation(text, clf, tfidf, num_features=12):
    explainer = LimeTextExplainer(class_names=['Fake News', 'Real News'])

    def predict_fn(texts):
        processed = [preprocess(t) for t in texts]
        X = tfidf.transform(processed)
        return clf.predict_proba(X)

    exp = explainer.explain_instance(
        text, predict_fn,
        num_features=num_features,
        num_samples=800
    )
    return exp

# ─────────────────────────────────────────────────────────────────────────────
# NEWSAPI  (hardcoded key, no user input needed)
# ─────────────────────────────────────────────────────────────────────────────
def search_live_news(query: str):
    """Cross-check text against live NewsAPI results."""
    try:
        key_terms = ' '.join(query.split()[:8])
        params = {
            'q': key_terms,
            'sortBy': 'relevancy',
            'pageSize': 4,
            'language': 'en',
            'apiKey': NEWSAPI_KEY,
        }
        r = requests.get(NEWSAPI_URL, params=params, timeout=8)
        if r.status_code == 200:
            return r.json().get('articles', [])
    except Exception:
        pass
    return []

def fetch_top_headlines(category: str = 'general'):
    """Fetch top headlines for the Live Headlines tab."""
    try:
        params = {
            'country': 'us',
            'category': category,
            'pageSize': 6,
            'apiKey': NEWSAPI_KEY,
        }
        r = requests.get(NEWSAPI_HEADLINES, params=params, timeout=8)
        if r.status_code == 200:
            return r.json().get('articles', [])
    except Exception:
        pass
    return []

# ─────────────────────────────────────────────────────────────────────────────
# URL SCRAPER
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_url(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; FakeNewsBot/2.0)'}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, 'html.parser')
        for tag in soup(['script','style','nav','footer','header','aside','form']):
            tag.decompose()
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ''
        paras = soup.find_all('p')
        body = ' '.join(p.get_text().strip() for p in paras if len(p.get_text()) > 40)
        return f"{title_text}\n\n{body}"[:6000]
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZER
# ─────────────────────────────────────────────────────────────────────────────
def summarize_text(text: str, n: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return ' '.join(sentences[:n]) if sentences else text[:300]

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def gauge_chart(confidence, pred):
    color = "#22d3a0" if pred == 1 else "#ff4d6d"
    if confidence < 52:
        color = "#fbbf24"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text':"Confidence Score",'font':{'size':15,'color':'#8b8fa8','family':'Sora'}},
        number={'font':{'size':46,'color':'#e8eaf0','family':'Sora'}},
        gauge={
            'axis':{'range':[0,100],'tickfont':{'color':'#8b8fa8','size':9}},
            'bar':{'color':color,'thickness':0.25},
            'bgcolor':"#1a1d27",'borderwidth':0,
            'steps':[
                {'range':[0,40],'color':'#2d1b1b'},
                {'range':[40,60],'color':'#2d2a1b'},
                {'range':[60,80],'color':'#1b2d24'},
                {'range':[80,100],'color':'#0f2d1f'},
            ],
            'threshold':{'line':{'color':color,'width':3},'thickness':0.75,'value':confidence}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=280, margin=dict(l=20,r=20,t=40,b=20), font={'family':'Sora'}
    )
    return fig

def lime_chart(lime_exp):
    feats   = lime_exp.as_list()
    words   = [f[0] for f in feats]
    weights = [f[1] for f in feats]
    colors  = ['#22d3a0' if w > 0 else '#ff4d6d' for w in weights]
    fig = go.Figure(go.Bar(
        x=weights, y=words, orientation='h',
        marker_color=colors, marker_line_width=0, opacity=0.85,
        text=[f"{w:+.4f}" for w in weights],
        textposition='outside',
        textfont={'size':9,'color':'#8b8fa8','family':'JetBrains Mono'}
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=360, margin=dict(l=10,r=80,t=20,b=20),
        xaxis=dict(
            title=dict(text='Contribution to Prediction',
                       font=dict(color='#8b8fa8', size=11, family='Sora')),
            tickfont=dict(color='#8b8fa8', size=9, family='JetBrains Mono'),
            gridcolor='#2a2d3e', zerolinecolor='#3a3d4e'
        ),
        yaxis=dict(
            tickfont=dict(color='#e8eaf0', size=11, family='JetBrains Mono'),
            gridcolor='rgba(0,0,0,0)', autorange='reversed'
        ),
        font={'family':'Sora'}
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# VOTE STORE
# ─────────────────────────────────────────────────────────────────────────────
def save_vote(summary, pred, confidence, vote):
    if 'votes' not in st.session_state:
        st.session_state.votes = []
    st.session_state.votes.append({
        'summary': summary[:120],
        'prediction': 'Real' if pred == 1 else 'Fake',
        'confidence': round(confidence,1),
        'vote': vote,
        'time': datetime.now().strftime('%H:%M:%S')
    })

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # HERO
    st.markdown("""
    <div class="hero">
      <div class="hero-icon">🗞️</div>
      <h1>Fake News Detection App</h1>
      <p>Voting Ensemble ML · LIME Explainability · Real-Time NewsAPI Verification</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ─────────────────────────────────────────────────────────
    with st.spinner("🧠 Loading AI model — first run trains the model, subsequent runs use cache…"):
        clf, tfidf, acc, ds_size = load_or_train_model()

    # ── Metrics ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", f"{acc*100:.1f}%")
    c2.metric("Algorithm",      "Voting Ensemble")
    c3.metric("Training Samples", f"{ds_size:,}")
    c4.metric("Explainability", "LIME")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main tabs ──────────────────────────────────────────────────────────
    main_tab1, main_tab2, main_tab3 = st.tabs(["🔍 Analyze News", "📰 Live Headlines", "📊 Vote History"])

    # ══════════════════════════════════════════════════════════════════════
    with main_tab1:
        st.markdown("## 📝 Enter News Content")
        inp_tab1, inp_tab2 = st.tabs(["✍️ Paste Text", "🔗 Enter URL"])

        input_text = None
        with inp_tab1:
            raw = st.text_area(
                "", height=180,
                placeholder="Paste any news article text here and click Analyze…",
                label_visibility="collapsed"
            )
            if raw:
                input_text = raw

        with inp_tab2:
            url_in = st.text_input(
                "", placeholder="https://example.com/news-article",
                label_visibility="collapsed"
            )
            if url_in:
                with st.spinner("🌐 Fetching article…"):
                    extracted = extract_text_from_url(url_in)
                if extracted and len(extracted) > 100:
                    input_text = extracted
                    st.success(f"✅ Extracted {len(extracted):,} characters")
                    with st.expander("Preview extracted text"):
                        st.write(extracted[:600] + "…")
                else:
                    st.error("❌ Could not extract text. Try pasting directly.")

        if st.button("🔍 Analyze News", use_container_width=True):
            if not input_text or len(input_text.strip()) < 15:
                st.warning("⚠️ Please enter at least a sentence of news text.")
            else:
                st.markdown("---")

                with st.spinner("🤖 Running ML analysis + LIME…"):
                    pred, confidence, proba = predict_news(input_text, clf, tfidf)
                    summary  = summarize_text(input_text)
                    lime_exp = get_lime_explanation(input_text, clf, tfidf)

                # ── Summary ───────────────────────────────────────────────
                st.markdown("### 📋 Summary")
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

                # ── Verdict ───────────────────────────────────────────────
                st.markdown("### 🔎 Prediction Result")
                if confidence < 52:
                    st.markdown('<div class="result-uncertain">⚠️ UNCERTAIN — verify manually</div>', unsafe_allow_html=True)
                elif pred == 1:
                    st.markdown('<div class="result-real">✅ Real News</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-fake">❌ Fake News</div>', unsafe_allow_html=True)

                # ── Gauge + probability bars ───────────────────────────────
                cg1, cg2 = st.columns([1.3, 0.7])
                with cg1:
                    st.plotly_chart(gauge_chart(confidence, pred),
                                    use_container_width=True,
                                    config={'displayModeBar': False})
                with cg2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Probability Breakdown</div>', unsafe_allow_html=True)
                    real_p = proba[1]*100
                    fake_p = proba[0]*100
                    st.markdown(f"""
                    <div style="margin:.8rem 0">
                      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="color:#22d3a0;font-weight:600;font-size:.9rem">✅ Real</span>
                        <span style="color:#22d3a0;font-family:'JetBrains Mono';font-size:.9rem">{real_p:.1f}%</span>
                      </div>
                      <div style="background:#2a2d3e;border-radius:6px;height:10px;overflow:hidden">
                        <div style="width:{real_p}%;background:linear-gradient(90deg,#22d3a0,#34d399);height:100%;border-radius:6px"></div>
                      </div>
                    </div>
                    <div style="margin:.8rem 0">
                      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="color:#ff4d6d;font-weight:600;font-size:.9rem">❌ Fake</span>
                        <span style="color:#ff4d6d;font-family:'JetBrains Mono';font-size:.9rem">{fake_p:.1f}%</span>
                      </div>
                      <div style="background:#2a2d3e;border-radius:6px;height:10px;overflow:hidden">
                        <div style="width:{fake_p}%;background:linear-gradient(90deg,#ff4d6d,#fb7185);height:100%;border-radius:6px"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    verdict = ("🟢 High confidence" if confidence >= 75
                               else "🟡 Medium confidence" if confidence >= 52
                               else "🔴 Low — verify manually")
                    st.markdown(f'<div style="color:#8b8fa8;font-size:.85rem;margin-top:1rem">{verdict}</div>', unsafe_allow_html=True)

                # ── LIME ──────────────────────────────────────────────────
                st.markdown("### 🔬 LIME Feature Importance")
                st.markdown('<div style="color:#8b8fa8;font-size:.88rem;margin-bottom:1rem">Words driving the prediction. <span style="color:#22d3a0">Green → Real</span> · <span style="color:#ff4d6d">Red → Fake</span></div>', unsafe_allow_html=True)
                st.plotly_chart(lime_chart(lime_exp), use_container_width=True,
                                config={'displayModeBar': False})

                feats = lime_exp.as_list()
                cl1, cl2 = st.columns(2)
                with cl1:
                    st.markdown('<div class="section-label">🟢 Real Indicators</div>', unsafe_allow_html=True)
                    for w, v in feats:
                        if v > 0:
                            st.markdown(f'<div class="lime-positive">+{v:.4f} &nbsp; {w}</div>', unsafe_allow_html=True)
                with cl2:
                    st.markdown('<div class="section-label">🔴 Fake Indicators</div>', unsafe_allow_html=True)
                    for w, v in feats:
                        if v < 0:
                            st.markdown(f'<div class="lime-negative">{v:.4f} &nbsp; {w}</div>', unsafe_allow_html=True)

                # ── NewsAPI cross-check ───────────────────────────────────
                st.markdown("### 🌐 Real-Time News Cross-Check")
                with st.spinner("🔍 Searching live news sources…"):
                    articles = search_live_news(input_text)

                if articles:
                    st.markdown(f'<div style="color:#22d3a0;font-size:.9rem;margin-bottom:.8rem">✅ Found {len(articles)} related articles from verified sources</div>', unsafe_allow_html=True)
                    for art in articles:
                        t  = art.get('title','—')
                        src= art.get('source',{}).get('name','Unknown')
                        u  = art.get('url','#')
                        dt = art.get('publishedAt','')[:10]
                        st.markdown(f"""
                        <div class="news-match">
                          <div style="font-weight:600;color:#e8eaf0;margin-bottom:4px">{t}</div>
                          <div style="color:#8b8fa8;font-size:.8rem">📰 {src} · 📅 {dt} · <a href="{u}" target="_blank" style="color:#4f8ef7">Read →</a></div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="news-no-match">
                      <div style="color:#fbbf24;font-weight:600">⚠️ No matching articles found in live news</div>
                      <div style="color:#8b8fa8;font-size:.82rem;margin-top:4px">Absence of coverage may indicate misinformation. Cross-check trusted sources.</div>
                    </div>""", unsafe_allow_html=True)

                # ── Voting ────────────────────────────────────────────────
                st.markdown("### 🗳️ Do you agree with this prediction?")
                cv1, cv2, cv3 = st.columns([1,1,3])
                with cv1:
                    if st.button("👍 Agree", use_container_width=True):
                        save_vote(summary, pred, confidence, "agree")
                        st.success("✅ Thanks for your feedback!")
                with cv2:
                    if st.button("👎 Disagree", use_container_width=True):
                        save_vote(summary, pred, confidence, "disagree")
                        st.warning("📝 Noted! Your feedback helps improve accuracy.")

    # ══════════════════════════════════════════════════════════════════════
    with main_tab2:
        st.markdown("## 📰 Live Top Headlines")
        cat = st.selectbox("Category", ["general","business","technology","health","science","sports","entertainment"])
        if st.button("🔄 Fetch Headlines", use_container_width=False):
            with st.spinner("📡 Fetching live headlines…"):
                headlines = fetch_top_headlines(cat)
            if headlines:
                for art in headlines:
                    title   = art.get('title','—')
                    source  = art.get('source',{}).get('name','Unknown')
                    url     = art.get('url','#')
                    desc    = art.get('description') or ''
                    pub     = art.get('publishedAt','')[:10]
                    full_t  = f"{title}. {desc}"

                    # Quick ML prediction on headline
                    p, conf, _ = predict_news(full_t, clf, tfidf)
                    badge = "✅ Real" if (p==1 and conf>=52) else ("❌ Fake" if (p==0 and conf>=52) else "⚠️ Uncertain")
                    bcolor= "#22d3a0" if p==1 else ("#ff4d6d" if conf>=52 else "#fbbf24")
                    st.markdown(f"""
                    <div class="news-match" style="margin-bottom:.8rem">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <div style="font-weight:600;color:#e8eaf0;font-size:.9rem;flex:1;margin-right:1rem">{title}</div>
                        <div style="color:{bcolor};font-weight:700;font-size:.85rem;white-space:nowrap">{badge} {conf:.0f}%</div>
                      </div>
                      <div style="color:#8b8fa8;font-size:.78rem;margin-top:4px">📰 {source} · 📅 {pub} · <a href="{url}" target="_blank" style="color:#4f8ef7">Read →</a></div>
                      {f'<div style="color:#8b8fa8;font-size:.82rem;margin-top:6px">{desc[:120]}…</div>' if desc else ''}
                    </div>""", unsafe_allow_html=True)
            else:
                st.error("❌ Could not fetch headlines. Check your internet connection.")

    # ══════════════════════════════════════════════════════════════════════
    with main_tab3:
        st.markdown("## 📊 Your Vote History")
        votes = st.session_state.get('votes', [])
        if votes:
            df_v = pd.DataFrame(votes)
            st.dataframe(df_v, use_container_width=True)
            agree_n    = sum(1 for v in votes if v['vote']=='agree')
            disagree_n = len(votes) - agree_n
            cv1, cv2, cv3 = st.columns(3)
            cv1.metric("Total Votes",  len(votes))
            cv2.metric("Agreed",       agree_n)
            cv3.metric("Disagreed",    disagree_n)
        else:
            st.info("No votes recorded yet. Analyze some articles and cast your vote!")

    # FOOTER
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#8b8fa8;font-size:.8rem;padding:1rem 0">
      🗞️ Fake News Detection App · Voting Ensemble (LR + RF + GBM) · LIME · NewsAPI<br>
      <span style="font-size:.73rem">⚠️ AI predictions are probabilistic. Always verify from multiple trusted sources.</span>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()