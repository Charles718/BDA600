# walmart_pipeline_mlflow_v16.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from itertools import product
import platform, sys, json
import plotly.express as px


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Walmart_Full_Clustering_Pipeline_v16"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ===================================================================
# MLflow Python Model Wrapper
# ===================================================================
class WalmartClusterModel(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler, kmeans):
        self.scaler = scaler
        self.kmeans = kmeans

    def predict(self, context, model_input):
        X_scaled = self.scaler.transform(model_input)
        return self.kmeans.predict(X_scaled)


# ===================================================================
# START MLflow RUN
# ===================================================================
with mlflow.start_run():

    # -----------------------
    # 0. Log environment info
    # -----------------------
    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("platform", platform.platform())
    mlflow.log_param("mlflow_version", mlflow.__version__)


# -----------------------------
# 1. Load & Clean Walmart Data
# -----------------------------
df = pd.read_csv("Walmart.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date"])

# Load and apply metadata
with open("Walmart_metadata.json", "r") as f:
    metadata = json.load(f)

for col, dtype in metadata["dtypes"].items():
    if dtype == 'int64':
        df[col] = df[col].astype('Int64')
    elif dtype == 'category':
        df[col] = df[col].astype('category')
    elif dtype == 'float64':
        df[col] = df[col].astype('float')
    elif dtype == 'bool':
        df[col] = df[col].astype('bool')
    elif dtype == 'object':
        df[col] = df[col].astype('object')

# Compute store-level features
store_features = df.groupby("Store").agg({
    "Temperature": "mean",
    "Fuel_Price": "mean",
    "CPI": "mean",
    "Unemployment": "mean"
}).reset_index()

# -----------------------------
# 2. K-Means Clustering
# -----------------------------
X = store_features[["Temperature", "Fuel_Price", "CPI", "Unemployment"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
store_features["Cluster"] = kmeans.fit_predict(X_scaled)

# Extract centroids in original scale
centroids_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=["Temperature", "Fuel_Price", "CPI", "Unemployment"]
)
centroids_df["Cluster"] = centroids_df.index
print("\nCluster Centroids:")
print(centroids_df)

# -----------------------------
# 3. PCA Visualization
# -----------------------------
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
store_features["PC1"] = pca_coords[:,0]
store_features["PC2"] = pca_coords[:,1]

plt.figure(figsize=(9,7))
sns.scatterplot(
    x="PC1", y="PC2", hue="Cluster", data=store_features,
    palette="viridis", s=120
)
for _, r in store_features.iterrows():
    plt.text(r["PC1"]+0.02, r["PC2"]+0.02, str(int(r["Store"])), fontsize=8)
plt.title("PCA Cluster Visualization (Stores)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

# Save the plot to a file
plt.savefig("store_region_map.png")  # Save the plot as PNG file

# -----------------------------
# 4. Heatmap of Cluster Centroids
# -----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(
    centroids_df.set_index("Cluster"),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Cluster Centroid Heatmap (Original Scale)")
plt.savefig("centroid_heatmap.png")  # Save the heatmap as PNG file


# -----------------------------
# 5. Radar Chart of Cluster Features
# -----------------------------
features = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
num_vars = len(features)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8,8))
for idx, row in centroids_df.iterrows():
    values = row[features].tolist()
    values += values[:1]
    plt.polar(angles, values, label=f"Cluster {int(row['Cluster'])}")
    plt.fill(angles, values, alpha=0.1)
plt.xticks(angles[:-1], features)
plt.title("Cluster Feature Radar Chart")
plt.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
plt.show()

# -----------------------------
# 6. FRED API Setup
# -----------------------------
fred = Fred(api_key="3b0e871d2391d061ef929dff968373dc")  # <-- replace with your key

state_series = {
    "AL":"ALUR","AK":"AKUR","AZ":"AZUR","AR":"ARUR","CA":"CAUR","CO":"COUR",
    "CT":"CTUR","DE":"DEUR","FL":"FLUR","GA":"GAUR","HI":"HIUR","ID":"IDUR",
    "IL":"ILUR","IN":"INUR","IA":"IAUR","KS":"KSUR","KY":"KYUR","LA":"LAUR",
    "ME":"MEUR","MD":"MDUR","MA":"MAUR","MI":"MIUR","MN":"MNUR","MS":"MSUR",
    "MO":"MOUR","MT":"MTUR","NE":"NEUR","NV":"NVUR","NH":"NHUR","NJ":"NJUR",
    "NM":"NMUR","NY":"NYUR","NC":"NCUR","ND":"NDUR","OH":"OHUR","OK":"OKUR",
    "OR":"ORUR","PA":"PAUR","RI":"RIUR","SC":"SCUR","SD":"SDUR","TN":"TNUR",
    "TX":"TXUR","UT":"UTUR","VT":"VTUR","VA":"VAUR","WA":"WAUR","WV":"WVUR",
    "WI":"WIUR","WY":"WYUR"
}

cpi_series = {
    "National":"CPIAUCSL","New York-NJ-PA":"CUURA101SA0","Boston-MA-NH":"CUURA103SA0",
    "Chicago-IL-IN-WI":"CUURA207SA0","Detroit-MI":"CUURA208SA0","Dallas-Fort Worth-TX":"CUURA316SA0",
    "Houston-TX":"CUURA318SA0","Miami-FL":"CUURA320SA0","Atlanta-GA":"CUURA319SA0",
    "San Diego-CA":"CUUSA424SA0","Seattle-WA":"CUURA423SA0","Los Angeles-CA":"CUURA421SA0",
    "Washington-DC-MD-VA-WV":"CUURA311SA0"
}

# -----------------------------
# 7. Fetch FRED series
# -----------------------------
def fetch_fred_series(code):
    s = fred.get_series(
        code,
        observation_start=df["Date"].min(),
        observation_end=df["Date"].max()
    )
    s.index = pd.to_datetime(s.index)
    return s.resample("W").mean()

unemp_df = pd.DataFrame({st: fetch_fred_series(code) for st, code in state_series.items()})
cpi_df   = pd.DataFrame({rg: fetch_fred_series(code) for rg, code in cpi_series.items()})

state_stats = pd.DataFrame({
    "State": unemp_df.columns,
    "Avg_Unemp": unemp_df.mean()
}).reset_index(drop=True)

region_stats = pd.DataFrame({
    "Region": cpi_df.columns,
    "Avg_CPI": cpi_df.mean()
}).reset_index(drop=True)

# -----------------------------
# 8. Match Clusters to FRED Regions
# -----------------------------
candidates = pd.DataFrame(list(product(state_stats.State, region_stats.Region)),
                          columns=["State","Region"])
candidates = candidates.merge(state_stats, on="State")
candidates = candidates.merge(region_stats, on="Region")

cluster_matches = []
for _, row in centroids_df.iterrows():
    cluster_id = int(row["Cluster"])
    c_temp, c_fuel, c_cpi, c_unemp = row[features[0]], row[features[1]], row[features[2]], row[features[3]]

    candidates["cpi_diff"] = (candidates["Avg_CPI"] - c_cpi).abs()
    candidates["unemp_diff"] = (candidates["Avg_Unemp"] - c_unemp).abs()
    candidates["cpi_norm"] = candidates["cpi_diff"] / candidates["cpi_diff"].max()
    candidates["unemp_norm"] = candidates["unemp_diff"] / candidates["unemp_diff"].max()
    candidates["score"] = candidates["cpi_norm"] + candidates["unemp_norm"]

    best = candidates.loc[candidates["score"].idxmin()]

    cluster_matches.append({
        "Cluster": cluster_id,
        "Best_State": best["State"],
        "Best_Region": best["Region"],
        "Cluster_CPI": c_cpi,
        "Cluster_Unemp": c_unemp,
        "Match_Difference": best["score"]
    })

cluster_match_df = pd.DataFrame(cluster_matches)
print("\nCluster → Region Match:")
print(cluster_match_df)

# -----------------------------
# 9. Merge back to Stores
# -----------------------------
store_final = store_features.merge(cluster_match_df, on="Cluster")
store_final.to_csv("store_region_assignments.csv", index=False)
print("\nFinal Store Assignments Saved!")

# -----------------------------
# 10. U.S. Map Visualization (Option 2)
# -----------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load US states GeoJSON (individual state geometries)
url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
states = gpd.read_file(url)

# 2. Project to Albers Equal Area for accurate centroids
states = states.to_crs(epsg=5070)
states['lon'] = states.geometry.centroid.x
states['lat'] = states.geometry.centroid.y

# 3. Map store Best_State to full state names
state_abbrev_to_name = {
    'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California',
    'CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia',
    'HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa',
    'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland',
    'MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri',
    'MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey',
    'NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio',
    'OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina',
    'SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia',
    'WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming','DC':'District of Columbia'
}

store_locations = store_final.copy()
store_locations['State_Full'] = store_locations['Best_State'].map(state_abbrev_to_name)

# 4. Merge store data with state centroids
store_locations = store_locations.merge(
    states[['name','lon','lat']], left_on='State_Full', right_on='name', how='left'
)

# Check for stores with missing coordinates
missing = store_locations[store_locations['lon'].isna()]
if not missing.empty:
    print("Warning: Stores not matched to a state centroid:\n", missing[['Store','Best_State','State_Full']])

# 5. Plot the map with clusters
fig, ax = plt.subplots(1,1, figsize=(15,10))
states.plot(ax=ax, color='lightgrey', edgecolor='white')

clusters = store_locations['Cluster'].unique()
colors = plt.cm.tab10(range(len(clusters)))

for c, color in zip(clusters, colors):
    subset = store_locations[store_locations['Cluster']==c].copy()
    # Add small jitter to avoid overlapping points
    subset['lon'] += np.random.uniform(-50000,50000,len(subset))
    subset['lat'] += np.random.uniform(-50000,50000,len(subset))
    ax.scatter(subset['lon'], subset['lat'], color=color, s=150, label=f'Cluster {c}', alpha=0.7, edgecolor='k')

plt.title("Walmart Store Clusters Mapped to U.S. States", fontsize=18)
plt.legend(title="Cluster", fontsize=12)
plt.axis('off')
plt.show()

import plotly.express as px
import pandas as pd



# Save the figure to an HTML file
fig.write_html("plotly_interactive_plot.html")


# Example: match_df should have at least: "Store", "Best_State", "Match_Difference"
# If Match_Difference is missing, we just ignore it for coloring

# Aggregate by Best_State
state_counts = store_final.groupby("Best_State").agg(
    Num_Stores=("Store", "count"),
    Stores=("Store", lambda x: ", ".join(map(str, x))),
    Avg_Match_Diff=("Match_Difference", "mean")
).reset_index()


# Use Avg_Match_Diff if available, otherwise fill with a constant for coloring
if state_counts['Avg_Match_Diff'].isnull().all():
    state_counts['Avg_Match_Diff'] = 1  # constant color

# Create bubble map
fig = px.scatter_geo(
    state_counts,
    locations="Best_State",
    locationmode="USA-states",
    size="Num_Stores",
    color="Avg_Match_Diff",
    hover_name="Best_State",
    hover_data={"Num_Stores": True, "Stores": True, "Avg_Match_Diff": True},
    scope="usa",
    color_continuous_scale="Viridis_r",
    size_max=40  # maximum bubble size
)

# Layout tweaks
fig.update_layout(
    title_text="Walmart Stores: Best-State Matches (Bubble Size = Num Stores)",
    geo=dict(
        lakecolor="LightBlue",
        showlakes=True
    )
)

# Save the figure to an HTML file
fig.write_html("Walmart_store_bubble_map.html")

fig.show()
































# -----------------------------
# 5. FRED API Setup
# -----------------------------
fred = Fred(api_key="3b0e871d2391d061ef929dff968373dc")  # <-- replace with your key

state_series = {
    "AL":"ALUR","AK":"AKUR","AZ":"AZUR","AR":"ARUR","CA":"CAUR","CO":"COUR",
    "CT":"CTUR","DE":"DEUR","FL":"FLUR","GA":"GAUR","HI":"HIUR","ID":"IDUR",
    "IL":"ILUR","IN":"INUR","IA":"IAUR","KS":"KSUR","KY":"KYUR","LA":"LAUR",
    "ME":"MEUR","MD":"MDUR","MA":"MAUR","MI":"MIUR","MN":"MNUR","MS":"MSUR",
    "MO":"MOUR","MT":"MTUR","NE":"NEUR","NV":"NVUR","NH":"NHUR","NJ":"NJUR",
    "NM":"NMUR","NY":"NYUR","NC":"NCUR","ND":"NDUR","OH":"OHUR","OK":"OKUR",
    "OR":"ORUR","PA":"PAUR","RI":"RIUR","SC":"SCUR","SD":"SDUR","TN":"TNUR",
    "TX":"TXUR","UT":"UTUR","VT":"VTUR","VA":"VAUR","WA":"WAUR","WV":"WVUR",
    "WI":"WIUR","WY":"WYUR"
}

def fetch_fred_series(code):
    s = fred.get_series(
        code,
        observation_start=df["Date"].min(),
        observation_end=df["Date"].max()
    )
    s.index = pd.to_datetime(s.index)
    return s.resample("W").mean()

unemp_df = pd.DataFrame({st: fetch_fred_series(code) for st, code in state_series.items()})
state_stats = pd.DataFrame({
    "State": unemp_df.columns,
    "Avg_Unemp": unemp_df.mean()
}).reset_index(drop=True)

# -----------------------------
# 6. Match Clusters to FRED Regions
# -----------------------------
# Create dummy region matching for clusters
cluster_matches = []
for _, row in centroids_df.iterrows():
    cluster_id = int(row["Cluster"])
    c_temp, c_fuel, c_cpi, c_unemp = row[["Temperature", "Fuel_Price", "CPI", "Unemployment"]]

    cluster_matches.append({
        "Cluster": cluster_id,
        "Best_State": "TX",  # Example, replace with actual region matching logic
        "Best_Region": "Houston-TX",
        "Cluster_CPI": c_cpi,
        "Cluster_Unemp": c_unemp,
        "Match_Difference": 0.1  # Example, replace with actual matching score
    })

cluster_match_df = pd.DataFrame(cluster_matches)
print("\nCluster → Region Match:")
print(cluster_match_df)

# -----------------------------
# 7. Merge back to Stores
# -----------------------------
store_final = store_features.merge(cluster_match_df, on="Cluster")
store_final.to_csv("store_region_assignments.csv", index=False)
print("\nFinal Store Assignments Saved!")

# -----------------------------
# 8. U.S. Map Visualization (Optional)
# -----------------------------
# Assuming you've already plotted the U.S. map
plt.figure(figsize=(15, 10))
# [Map plotting code, e.g., US state boundaries]
plt.savefig("us_map_visualization.png")  # Save the map visualization

# -----------------------------
# 9. Save Plotly figure as HTML and Log as Artifact
# -----------------------------

# Example: match_df should have at least: "Store", "Best_State", "Match_Difference"
# If Match_Difference is missing, we just ignore it for coloring

# Aggregate by Best_State
state_counts = store_final.groupby("Best_State").agg(
    Num_Stores=("Store", "count"),
    Stores=("Store", lambda x: ", ".join(map(str, x))),
    Avg_Match_Diff=("Match_Difference", "mean")
).reset_index()


# Use Avg_Match_Diff if available, otherwise fill with a constant for coloring
if state_counts['Avg_Match_Diff'].isnull().all():
    state_counts['Avg_Match_Diff'] = 1  # constant color
# Create the Plotly map
fig = px.scatter_geo(
    state_counts,
    locations="Best_State",
    locationmode="USA-states",
    size="Num_Stores",
    color="Avg_Match_Diff",
    hover_name="Best_State",
    hover_data={"Num_Stores": True, "Stores": True, "Avg_Match_Diff": True},
    scope="usa",
    color_continuous_scale="Viridis_r",
    size_max=40  # maximum bubble size
)

# Layout tweaks for the figure
fig.update_layout(
    title_text="Walmart Stores: Best-State Matches (Bubble Size = Num Stores)",
    geo=dict(
        lakecolor="LightBlue",
        showlakes=True
    )
)

# Save the Plotly figure as an HTML file
fig.write_html("us_map_visualization.html")

# Log the HTML file as an artifact in MLflow
mlflow.log_artifact("us_map_visualization.html")

# -----------------------------
# 10. MLflow Logging (Model, Artifacts, etc.)
# -----------------------------
# Ensure the previous run is ended before starting a new one
if mlflow.active_run() is not None:
    mlflow.end_run()

# Start a new MLflow run
with mlflow.start_run():
    # Log parameters, metrics, etc.
    mlflow.log_param("n_clusters", 4)
    mlflow.log_param("scaling", "StandardScaler")
    
    # Log the model
    mlflow.sklearn.log_model(kmeans, "kmeans_model")

    # Log artifacts (images)
    mlflow.log_artifact("store_region_map.png")
    mlflow.log_artifact("centroid_heatmap.png")
    mlflow.log_artifact("us_map_visualization.png")

# -----------------------------
# 10. Conclusion
# -----------------------------
print("MLflow run completed and artifacts logged successfully!")

