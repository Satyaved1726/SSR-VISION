from io import BytesIO
from datetime import datetime
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def build_intelligence_pdf(session_state):
    """Generate a professional PDF intelligence report from session state."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#00B7D3"),
        fontSize=21,
        leading=24,
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "ReportSub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#243746"),
        leading=14,
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0B2D4D"),
        fontSize=13,
        leading=16,
        spaceBefore=6,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCopy",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#15202B"),
    )

    elements = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vision = session_state.get("vision_results", {})
    web_data = session_state.get("web_data", {})

    header_table = Table([
        [
            Paragraph("CYBER POLICE TRAFFIC DOSSIER", title_style),
            Paragraph(f"Generated: {now_str}<br/>Security Level: Internal Cyber Intelligence", sub_style),
        ]
    ], colWidths=[310, 220])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF7FB")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#00B7D3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 12))

    caption = session_state.get("caption", "No summary generated")
    fusion = session_state.get("fusion_insight", "No fusion insight generated")
    elements.append(Paragraph("Executive Summary", section_style))
    elements.append(Paragraph(caption, body_style))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Fusion Insight: {fusion}", body_style))
    elements.append(Spacer(1, 12))

    metrics_table = Table([
        ["Metric", "Value"],
        ["Traffic Risk Score", str(session_state.get("risk_score", 0))],
        ["Vehicle Count", str(vision.get("vehicle_count", 0))],
        ["Pedestrian Count", str(vision.get("pedestrian_count", 0))],
        ["Traffic Density", str(vision.get("density_level", "LOW"))],
        ["Lane Occupancy", str(vision.get("lane_occupancy", 0.0))],
        ["Vehicle Spacing", str(vision.get("vehicle_spacing", 0.0))],
        ["Road Condition", str(vision.get("road_condition", "N/A"))],
        ["Detected Violations", str(len(vision.get("violations", [])))],
        ["Detected Plates", str(len(session_state.get("plates", [])))],
        ["Weather Insight", str(web_data.get("weather", "Unknown"))],
    ], colWidths=[180, 320])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2D4D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A0A0A0")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F6F9FC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F6F9FC"), colors.HexColor("#ECF4F8")]),
    ]))

    elements.append(Paragraph("Operational Metrics", section_style))
    elements.append(metrics_table)
    elements.append(Spacer(1, 12))

    violations = vision.get("violations", [])
    elements.append(Paragraph("Violation Log", section_style))
    if violations:
        for v in violations:
            elements.append(Paragraph(f"- {v}", body_style))
    else:
        elements.append(Paragraph("- No major violations detected", body_style))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Violation Statistics", section_style))
    if violations:
        violation_counts = Counter(v.split(" ")[0] for v in violations)
        for key, count in violation_counts.items():
            elements.append(Paragraph(f"- {key}: {count}", body_style))
    else:
        elements.append(Paragraph("- No violation classes to summarize", body_style))

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Vehicle Intelligence", section_style))
    vehicles = vision.get("vehicles", [])
    if vehicles:
        for idx, v in enumerate(vehicles[:10], start=1):
            line = (
                f"- #{idx}: {v.get('vehicle_type', 'Vehicle')} | Color: {v.get('vehicle_color', 'Unknown')} "
                f"| Model: {v.get('vehicle_model', 'Unknown')}"
            )
            elements.append(Paragraph(line, body_style))
    else:
        elements.append(Paragraph("- No vehicles detected", body_style))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Number Plate Recognition", section_style))
    plates = session_state.get("plates", [])
    if plates:
        elements.append(Paragraph("- " + ", ".join(plates), body_style))
    else:
        elements.append(Paragraph("- No plate extracted", body_style))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Weather and Web Intelligence", section_style))
    advisories = web_data.get("advisories", [])
    if advisories:
        for item in advisories:
            elements.append(Paragraph(f"- {item}", body_style))
    else:
        elements.append(Paragraph("- No active advisories", body_style))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"- Weather Insight: {str(web_data.get('weather', 'Unknown'))}", body_style))

    entities = web_data.get("entities", {})
    locations = entities.get("locations", []) if isinstance(entities, dict) else []
    conditions = entities.get("conditions", []) if isinstance(entities, dict) else []
    if locations or conditions:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Text Intelligence Entities", section_style))
        elements.append(Paragraph(f"- Locations: {', '.join(locations) if locations else 'None'}", body_style))
        elements.append(Paragraph(f"- Conditions: {', '.join(conditions) if conditions else 'None'}", body_style))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
