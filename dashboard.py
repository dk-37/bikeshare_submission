import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
sns.set(style="dark")

# Menyiapkan helper function
# create_daily_rent_df() untuk menyiapkan  daily_rent_df
def create_daily_rent_df(df):
    daily_rent_df = df.resample(rule="D", on="dteday").agg({
        "cnt": "sum"
    })
    daily_rent_df = daily_rent_df.reset_index()
    daily_rent_df.columns = ["date", "rent_count"]
    
    return daily_rent_df

# create_bymembership_df() untuk menyiapkan bymembership_df
def create_bymembership_df(df):
    bymembership_df = df[["dteday", "casual", "registered"]] # slicing tabel day_df
    bymembership_df.set_index("dteday", inplace=True)
    bymembership_df = bymembership_df.stack().reset_index() # reshaping tabel bymembership_df
    bymembership_df.columns = ["date", "membership", "rent_count"] # assign nama kolom
    bymembership_df = bymembership_df.groupby(by="membership")["rent_count"].sum().reset_index() # agregasi berdasarkan kolom membership
    
    return bymembership_df

# create_environmental_df() untuk menyiapkan environmental_df
def create_environmental_df(df):
    environmental_df = df[["season", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"]]
    return environmental_df

# create_detection_df() untuk menyiapkan detection_df
def create_detection_df(df):
    mean = df["cnt"].mean()
    std = df["cnt"].std()
    df["z_score"] = df["cnt"].apply(lambda x: (x - mean)/std)
    df["isOutlier"] = df["z_score"].apply(lambda x: "Not Outlier" if -3 < x < 3 else "Outlier")
    detection_df = df[["dteday", "cnt", "z_score", "isOutlier"]]
    detection_df.columns = ["date", "rent_count", "z_score", "isOutlier"]
    
    return detection_df

# Memuat berkas all_day.csv
all_day_df = pd.read_csv("all_day.csv")

# Mengurutkan data berdasarkan dteday
all_day_df.sort_values(by="dteday", inplace=True)
all_day_df.reset_index(inplace=True)
all_day_df["dteday"] = pd.to_datetime(all_day_df["dteday"])

# Membuat komponen filter
min_date = all_day_df["dteday"].min()
max_date = all_day_df["dteday"].max()

with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("bikeshare_02.png")
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Periode',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_day_df[(all_day_df["dteday"] >= str(start_date)) & (all_day_df["dteday"] <= str(end_date))]

daily_rent_df = create_daily_rent_df(main_df)
bymembership_df = create_bymembership_df(main_df)
environmental_df = create_environmental_df(main_df)
detection_df = create_detection_df(main_df)
outlier_df = detection_df[detection_df["isOutlier"] == ("Outlier")]

st.header("BikeShare Dashboard")

st.subheader('Daily Rent')

col1, col2, col3 = st.columns(3)

with col1:
    total_rent = daily_rent_df["rent_count"].sum()
    st.metric("Total rent", value=total_rent)

with col2:
    registered = bymembership_df[bymembership_df["membership"] == "registered"]["rent_count"]
    st.metric("Registered rent", value=registered)

with col3:
    casual = bymembership_df[bymembership_df["membership"] == "casual"]["rent_count"]
    st.metric("Casual rent", value=casual)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_rent_df["date"],
    daily_rent_df["rent_count"],
    marker='o', 
    linewidth=2,
    color="#003B73"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
 
st.pyplot(fig)

st.subheader("Membership Segmentation")

colors=["#003B73", "#BFD7ED"]

fig = plt.figure(figsize=(10, 5)) 
sns.barplot(     
    x="membership",
    y="rent_count",
    data=bymembership_df.sort_values(by="rent_count", ascending=False),
    hue="membership",
    palette=colors
)
plt.ylabel(None)
plt.xlabel(None)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
st.pyplot(fig)

st.subheader("Environmental Variables Correlation")

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 30))

sns.stripplot(
    x="season", 
    y="cnt",
    data=environmental_df.sort_values(by="season", ascending=True),
    hue="season",
    palette=["#BFD7ED", "#60A3D9", "#0074B7", "#003B73"],
    legend=False,
    ax=ax[0, 0]
)
ax[0, 0].set_ylabel(None)
ax[0, 0].set_xlabel(None)
ax[0, 0].set_title("By Seasons", loc="center", fontsize=25)
ax[0, 0].tick_params(axis ='x', labelsize=18)
ax[0, 0].tick_params(axis ='y', labelsize=15)

sns.stripplot(
    x="weathersit", 
    y="cnt",
    data=environmental_df.sort_values(by="weathersit", ascending=True),
    hue="weathersit",
    palette=["#60A3D9", "#0074B7", "#003B73"],
    legend=False,
    ax=ax[0, 1]
)
ax[0, 1].set_ylabel(None)
ax[0, 1].set_xlabel(None)
ax[0, 1].set_title("By Weathersit", loc="center", fontsize=25)
ax[0, 1].tick_params(axis ='x', labelsize=18)
ax[0, 1].tick_params(axis ='y', labelsize=15)

sns.regplot(
    x="temp", 
    y="cnt",
    data=environmental_df.sort_values(by="temp", ascending=True),
    color="#003B73",
    ax=ax[1, 0]
)
ax[1, 0].set_ylabel(None)
ax[1, 0].set_xlabel(None)
ax[1, 0].set_title("By Temperature", loc="center", fontsize=25)
ax[1, 0].tick_params(axis ='x', labelsize=18)
ax[1, 0].tick_params(axis ='y', labelsize=15)

sns.regplot(
    x="atemp", 
    y="cnt",
    data=environmental_df.sort_values(by="atemp", ascending=True),
    color="#003B73",
    ax=ax[1, 1]
)
ax[1, 1].set_ylabel(None)
ax[1, 1].set_xlabel(None)
ax[1, 1].set_title("By Feeling Temperature", loc="center", fontsize=25)
ax[1, 1].tick_params(axis ='x', labelsize=18)
ax[1, 1].tick_params(axis ='y', labelsize=15)

sns.regplot(
    x="hum", 
    y="cnt",
    data=environmental_df.sort_values(by="hum", ascending=True), 
    color="#003B73",
    ax=ax[2, 0]
)
ax[2, 0].set_ylabel(None)
ax[2, 0].set_xlabel(None)
ax[2, 0].set_title("By Humidity", loc="center", fontsize=25)
ax[2, 0].tick_params(axis ='x', labelsize=18)
ax[2, 0].tick_params(axis ='y', labelsize=15)

sns.regplot(
    x="windspeed", 
    y="cnt",
    data=environmental_df.sort_values(by="windspeed", ascending=True),
    color="#003B73",
    ax=ax[2, 1]
)
ax[2, 1].set_ylabel(None)
ax[2, 1].set_xlabel(None)
ax[2, 1].set_title("By Windspeed", loc="center", fontsize=25)
ax[2, 1].tick_params(axis ='x', labelsize=18)
ax[2, 1].tick_params(axis ='y', labelsize=15)

st.pyplot(fig)

st.subheader("Anomaly Detection")

col1, col2 = st.columns(2)

with col1:
    if outlier_df.empty:
        st.metric("Lowest outlier", value="No outliers")
    else:
        outlier_min = outlier_df["rent_count"].min()
        st.metric("Lowest outlier", value=outlier_min)

with col2:
    if outlier_df.empty:
        st.metric("Highest outlier", value="No outliers")        
    else:
        outlier_max = outlier_df["rent_count"].max()
        st.metric("Highest outlier", value=outlier_max)

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    x="date", 
    y="z_score",
    data=detection_df,
    color="#003B73"
)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.tick_params(axis ='x', labelsize=12)
ax.tick_params(axis ='y', labelsize=12)
ax.hlines(y=[-3, 3], xmin=detection_df["date"].min(), xmax=detection_df["date"].max())
plt.xticks(rotation=45)
st.pyplot(fig)

st.caption('Copyright (c) BikeShare 2023')
