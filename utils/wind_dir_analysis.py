"""
Quick test to verify wind direction factors are correct.
Run this to check if the wind logic fix is working properly.
"""
import math

def test_wind_direction():
    """Test the corrected wind direction logic"""
    
    # FlamMap convention: wind_direction is direction wind blows FROM
    wind_from = 0.0  # Wind from North (default in your model)
    
    # Convert to wind TO direction
    wind_to = (wind_from + 180.0) % 360.0
    print(f"Wind FROM: {wind_from}° (North)")
    print(f"Wind TO: {wind_to}° (South)")
    print()
    
    # Test 8 neighbor directions
    directions = {
        'North':     0.0,
        'Northeast': 45.0,
        'East':      90.0,
        'Southeast': 135.0,
        'South':     180.0,
        'Southwest': 225.0,
        'West':      270.0,
        'Northwest': 315.0,
    }
    
    print("Fire Spread Direction -> Expected Factor -> Calculated Factor")
    print("-" * 70)
    
    for dir_name, heading_deg in directions.items():
        # Calculate angular difference
        diff = abs((heading_deg - wind_to + 180.0) % 360.0 - 180.0)
        
        # Determine factor
        if diff < 90.0:
            factor = 1.0
            fire_type = "HEAD"
        elif diff < 135.0:
            factor = 0.5
            fire_type = "FLANK"
        else:
            factor = 0.25
            fire_type = "BACK"
        
        # Expected factor based on wind
        if dir_name == "South":
            expected = "1.0 (HEAD)"
        elif dir_name == "Southeast" or dir_name == "Southwest":
            expected = "1.0 or 0.5 (HEAD/FLANK)"
        elif dir_name == "East" or dir_name == "West":
            expected = "0.5 (FLANK)"
        elif dir_name == "Northeast" or dir_name == "Northwest":
            expected = "0.5 or 0.25 (FLANK/BACK)"
        else:  # North
            expected = "0.25 (BACK)"
        
        print(f"{dir_name:12s} ({heading_deg:5.0f}°) -> {expected:20s} -> "
              f"{factor:.2f} ({fire_type})")
    
    print()
    print("✓ If Southeast shows 1.0 (HEAD) and Northwest shows 0.25 (BACK), the fix worked!")
    print("✗ If the opposite, there's still an issue.")

if __name__ == "__main__":
    test_wind_direction()