# ====================================================
# FILE: config_profiles.py
# ====================================================
# Central configuration file containing all profiles,
# their associated DOP raster tiles, and required Geofabrik regions.

PROFILES = {
    "example_bottrop": {
        "dop_tiles": ["355_5707", "356_5707", "356_5708", "357_5707", "357_5708"],
        "geofabrik_region": "duesseldorf-regbez",
    },
    "example_sonnberg": {
        "dop_tiles": ["366_5676", "366_5677"],
        "geofabrik_region": "duesseldorf-regbez",
    },
    "example_muenster": {
        "dop_tiles": [
            "402_5748",
            "402_5747",
            "402_5746",
            "401_5748",
            "401_5747",
            "401_5746",
        ],
        "geofabrik_region": "muenster-regbez",
    },
    "example_kamen": {
        "dop_tiles": [
            "407_5717",
            "408_5716",
            "408_5717",
            "409_5716",
            "409_5717",
            "409_5718",
        ],
        "geofabrik_region": "arnsberg-regbez",
    },
}
