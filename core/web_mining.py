import requests
from bs4 import BeautifulSoup
import random
import re

class WebDataMiner:
    def __init__(self, region="Hyderabad"):
        safe_region = (region or "Hyderabad").strip().replace(" ", "+")
        self.weather_url = f"https://wttr.in/{safe_region}?format=3"

    def mine_weather_and_traffic(self, region=None):
        if region:
            safe_region = region.strip().replace(" ", "+")
            self.weather_url = f"https://wttr.in/{safe_region}?format=3"
        data = {
            "weather": "Unknown",
            "traffic_news": "No latest updates.",
            "gov_alerts": [],
            "accident_reports": [],
            "is_bad_weather": False,
            "accident_locations": [],
            "road_closures": [],
            "advisories": [],
            "keywords": [],
            "entities": {
                "locations": [],
                "conditions": [],
            },
        }
        
        # 1. Weather Scraping
        try:
            response = requests.get(self.weather_url, timeout=5.0)
            if response.status_code == 200:
                weather_str = response.text.strip()
                data["weather"] = weather_str
                
                # Simple keyword heuristic for bad weather
                bad_weather_keywords = ["rain", "storm", "snow", "fog", "thunder", "cloud", "shower"]
                if any(kw in weather_str.lower() for kw in bad_weather_keywords):
                    data["is_bad_weather"] = True
        except Exception as e:
            print(f"Weather Mining Error: {e}")
            data["weather"] = "Data Unavailable (Offline)"

        # 2. Simulated Traffic News via BeautifulSoup (Mocked target for safety in production)
        # In a real app, you would scrape a local city's 511/traffic RSS feed.
        # Here we mock the BS4 parsing logic on a dummy HTML structure.
        dummy_html = """
        <html><body>
            <div class="news-item">Major congestion on Northern Highway due to construction.</div>
            <div class="news-item">Accident cleared on Main St, traffic flowing normally.</div>
            <div class="news-item">Government advisory: lane closure expected near Central Flyover tonight.</div>
            <div class="news-item">Heavy rain alert issued for metro corridor and ring road.</div>
        </body></html>
        """
        soup = BeautifulSoup(dummy_html, 'html.parser')
        news_items = soup.find_all('div', class_='news-item')
        if news_items:
            # Pick a random news item
            selected_news = random.choice(news_items).text
            data["traffic_news"] = selected_news
            all_news = [item.text.strip() for item in news_items]
            data["gov_alerts"] = [x for x in all_news if "advisory" in x.lower() or "alert" in x.lower()]
            data["accident_reports"] = [x for x in all_news if "accident" in x.lower()]

        data.update(self.analyze_text_intelligence(data))

        return data

    def analyze_text_intelligence(self, data):
        corpus = " ".join([
            data.get('weather', ''),
            data.get('traffic_news', ''),
            " ".join(data.get('gov_alerts', [])),
            " ".join(data.get('accident_reports', [])),
        ]).lower()
        keywords = [
            "rain", "storm", "fog", "accident", "closure", "congestion",
            "construction", "delay", "traffic", "advisory"
        ]
        detected = [word for word in keywords if word in corpus]

        accidents = re.findall(r"accident\s+(?:on|at)?\s*([a-zA-Z\s]+)", corpus)
        closures = []
        advisories = []

        if "construction" in corpus or "closure" in corpus:
            closures.append("Potential road closure or construction impact in active route")
        if "rain" in corpus or "storm" in corpus or "fog" in corpus:
            advisories.append("Weather advisory likely impacting braking distance and visibility")
        if "congestion" in corpus:
            advisories.append("Congestion advisory active from web intelligence feed")

        location_candidates = re.findall(r"(?:on|near|at)\s+([a-z\s]+(?:road|st|street|highway|flyover))", corpus)
        conditions = [c for c in ["rain", "storm", "fog", "construction", "closure"] if c in corpus]

        return {
            "keywords": detected,
            "accident_locations": [loc.strip() for loc in accidents if loc.strip()],
            "road_closures": closures,
            "advisories": advisories,
            "entities": {
                "locations": sorted(set([loc.strip().title() for loc in location_candidates if loc.strip()])),
                "conditions": sorted(set(conditions)),
            },
        }
