DEMO_VEHICLE_DB = {
    "TS09AB1234": {
        "owner_name": "A. Reddy",
        "vehicle_model": "Hyundai i20",
        "city": "Hyderabad",
    },
    "TS08CD5678": {
        "owner_name": "K. Sharma",
        "vehicle_model": "Maruti Swift",
        "city": "Secunderabad",
    },
    "TS07EF9012": {
        "owner_name": "R. Verma",
        "vehicle_model": "Tata Nexon",
        "city": "Warangal",
    },
}

PORTAL_LINKS = {
    "challan": "https://echallan.tspolice.gov.in/publicview/",
    "vahan": "https://vahan.parivahan.gov.in/vahanservice/vahan/ui/statevalidation/homepage.xhtml",
}


def lookup_owner_details(plate_number):
    if not plate_number:
        return None
    normalized = plate_number.replace(" ", "").upper()
    return DEMO_VEHICLE_DB.get(normalized)
