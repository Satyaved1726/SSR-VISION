class AnalyticsEngine:
    @staticmethod
    def calculate_risk_score(vision_results, web_data):
        """
        Calculates a Traffic Risk Score from 0 to 100 based on CV & Web Mining results.
        Includes rules for vehicle count, violations, road conditions, and weather.
        """
        score = 0
        
        # 1. Vehicle Count (Max 30 points)
        v_count = vision_results.get("vehicle_count", 0)
        score += min(v_count * 2, 30)

        # 1b. Pedestrian pressure in traffic corridor (Max 10 points)
        ped_count = vision_results.get("pedestrian_count", 0)
        score += min(ped_count * 2, 10)
        
        # 2. Violations (Max 30 points)
        v_violations = len(vision_results.get("violations", []))
        score += min(v_violations * 10, 30)

        # 3. Density (Max 15 points)
        density = vision_results.get("density_level", "LOW")
        if density == "CRITICAL":
            score += 15
        elif density == "HIGH":
            score += 10
        elif density == "MEDIUM":
            score += 5
        
        # 4. Road Condition (Max 15 points)
        road_cond = vision_results.get("road_condition", "GOOD")
        if road_cond == "DAMAGED/OBSTRUCTED":
            score += 15
        elif road_cond == "FADED MARKINGS":
            score += 5
            
        # 5. Weather and contextual advisories (Max 15 points)
        if web_data.get("is_bad_weather", False):
            score += 8
        score += min(len(web_data.get("advisories", [])) * 2, 5)
            
        return min(max(int(score), 0), 100) # Clamp 0-100

    @staticmethod
    def generate_insight_caption(vision_results, web_data, score):
        """
        Generates a natural language smart description of the analyzed image.
        """
        v_count = vision_results.get("vehicle_count", 0)
        p_count = vision_results.get("pedestrian_count", 0)
        density = vision_results.get("density_level", "LOW")
        v_violations = vision_results.get("violations", [])
        
        caption = f"A traffic scene with {v_count} vehicles and {p_count} pedestrians detected, indicating {density.lower()} congestion levels. "
        
        if v_violations:
            caption += f"AI identified {len(v_violations)} rule violation(s) including {v_violations[0].lower()}. "
        else:
            caption += "No immediate violations detected. "
            
        if web_data.get("is_bad_weather"):
            caption += f"Additionally, {web_data.get('weather')} conditions may negatively impact road safety. "

        if web_data.get("advisories"):
            caption += f"Web intelligence indicates: {web_data.get('advisories')[0]}. "
            
        if score > 75:
            caption += "Overall situation is CRITICAL, requiring immediate traffic management intervention."
        elif score > 40:
            caption += "Overall situation is MODERATE. Proceed with standard monitoring."
        else:
            caption += "Overall situation is SAFE."
            
        return caption

    @staticmethod
    def compute_traffic_metrics(vision_results):
        vehicle_count = vision_results.get("vehicle_count", 0)
        lane_occupancy = vision_results.get("lane_occupancy", 0.0)
        spacing = vision_results.get("vehicle_spacing", 0.0)
        violations = len(vision_results.get("violations", []))
        return {
            "vehicle_count": vehicle_count,
            "pedestrian_count": vision_results.get("pedestrian_count", 0),
            "lane_occupancy": lane_occupancy,
            "vehicle_spacing": spacing,
            "violation_frequency": violations,
            "distribution": vision_results.get("type_distribution", {}),
        }

    @staticmethod
    def fuse_intelligence(vision_results, web_data, score):
        if vision_results.get("density_level") in ["HIGH", "CRITICAL"] and web_data.get("is_bad_weather"):
            return "Rainfall or adverse weather is likely amplifying congestion observed in the scene."
        if vision_results.get("pedestrian_count", 0) > 5 and web_data.get("is_bad_weather"):
            return "Reduced visibility with high pedestrian presence indicates increased urban collision risk."
        if vision_results.get("violations") and web_data.get("advisories"):
            return "Field violations align with web advisories, indicating elevated urban traffic risk."
        if score > 70:
            return "Multi-source fusion indicates a high-risk corridor requiring active intervention."
        return "CVIP and WDM signals are consistent with normal monitoring conditions."

    @staticmethod
    def generate_alerts(vision_results, web_data, score):
        """
        Generates a list of actionable alerts based on thresholds.
        """
        alerts = []
        
        if score > 80:
            alerts.append({"level": "critical", "msg": "CRITICAL RISK SCORE EXCEEDED (80+)"})
            
        if vision_results.get("density_level") in ["HIGH", "CRITICAL"]:
            alerts.append({"level": "warning", "msg": "HIGH TRAFFIC DENSITY DETECTED"})
            
        for violation in vision_results.get("violations", []):
            alerts.append({"level": "critical", "msg": f"VIOLATION: {violation}"})
            
        if vision_results.get("road_condition") != "GOOD":
            alerts.append({"level": "warning", "msg": f"ROAD HAZARD: {vision_results.get('road_condition')}"})
            
        if web_data.get("is_bad_weather"):
            alerts.append({"level": "info", "msg": "WEATHER ADVISORY IN EFFECT FOR THIS ROUTE"})

        for advisory in web_data.get("advisories", []):
            alerts.append({"level": "info", "msg": f"WDM SIGNAL: {advisory}"})
            
        return alerts
