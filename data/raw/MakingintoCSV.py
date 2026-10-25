import glob
import geopandas as pd_geo
import pandas as pd

shapefiles = glob.glob("*.shp")

for shp in shapefiles:
    gdf = pd_geo.read_file(shp)
    base_name = shp.rsplit(".", 1)[0]

    gdf.to_parquet(f"{base_name}.parquet")

    gdf.to_file(f"{base_name}.geojson", driver="GeoJSON")

    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    df.to_csv(f"{base_name}_attributes.csv", index=False)