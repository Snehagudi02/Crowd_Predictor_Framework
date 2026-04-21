def get_risk_level(count):
    if count < 15:
        return "LOW"
    elif count <= 25:
        return "MODERATE"
    else:
        return "HIGH ALERT"
