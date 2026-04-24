power_map = {
    # Kitchen appliances
    "microwave": 1000,
    "oven": 1200,
    "refrigerator": 150,
    "toaster": 800,

    # Living room
    "tv": 150,
    "remote": 2,

    # Office / work setup
    "laptop": 65,
    "keyboard": 5,
    "mouse": 3,
    "cell phone": 10,

    # Audio / entertainment
    "speaker": 20,

    # Generic electronics (approximation)
    "book": 0,
    "cup": 0,
    "bottle": 0,
    "chair": 0,
    "sofa": 0,
    "bed": 0,

    # Approximation tricks (important)
    "tv monitor": 120,   # sometimes needed
}

def calculate_power(detected_objects):
    total_power = 0

    for obj in detected_objects:
        if obj in power_map:
            total_power += power_map[obj]

    return total_power

def calculate_cost(power, price_per_kwh=1500):
    kwh = power / 1000
    return kwh * price_per_kwh