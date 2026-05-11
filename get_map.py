import requests
import math
from PIL import Image
import io
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURATION — only change these values
# ─────────────────────────────────────────────

API_KEY = "AIzaSyAe3vPF7XLWs_H-_MOlfc79AaQRvl5GWNg"  # paste your key here

# Right-click your roundabout on Google Maps to get these
LATITUDE = 42.231433   # replace with your actual roundabout latitude
LONGITUDE = -82.991228 # replace with your actual roundabout longitude

ZOOM = 19            # 19 gives ~28cm per pixel — good for lane-level detail
IMAGE_SIZE = "640x640"  # max free tier size in pixels
MAP_TYPE = "satellite"  # satellite gives real aerial imagery

# ─────────────────────────────────────────────
# CALCULATE REAL-WORLD SCALE
# This tells you exactly what each pixel represents
# in metres — you'll use this number in your MPC later
# ─────────────────────────────────────────────

metres_per_pixel = (156543.03 * math.cos(math.radians(LATITUDE))) / (2 ** ZOOM)
print(f"Map scale: 1 pixel = {metres_per_pixel:.4f} metres")
print(f"Total image covers: {640 * metres_per_pixel:.1f} x {640 * metres_per_pixel:.1f} metres of real ground")

# ─────────────────────────────────────────────
# BUILD THE API REQUEST URL
# This is the "formatted request" you send to Google's server
# The URL is just a structured string with your parameters embedded
# ─────────────────────────────────────────────

url = (
    f"https://maps.googleapis.com/maps/api/staticmap?"
    f"center={LATITUDE},{LONGITUDE}"  # where to centre the image
    f"&zoom={ZOOM}"                   # how close to zoom in
    f"&size={IMAGE_SIZE}"             # pixel dimensions of returned image
    f"&maptype={MAP_TYPE}"            # satellite vs roadmap vs terrain
    f"&key={API_KEY}"                 # your authentication key
)

# ─────────────────────────────────────────────
# SEND THE REQUEST AND HANDLE THE RESPONSE
# requests.get() sends an HTTP GET request to the URL
# Google's server processes it and sends back the image as raw bytes
# ─────────────────────────────────────────────

print("Sending request to Google Maps API...")
response = requests.get(url)

# HTTP status code 200 means success
# Anything else means something went wrong
if response.status_code == 200:
    print("Success — image received")
    
    # Convert the raw bytes Google sent into an image object
    # io.BytesIO treats the bytes like a file without saving to disk first
    image = Image.open(io.BytesIO(response.content))
    
    # Save it permanently to your project folder
    image.save("roundabout_satellite.png")
    print("Image saved as roundabout_satellite.png")
    
    # Display it immediately so you can verify it looks right
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Roundabout Satellite Image\n"
              f"Scale: {metres_per_pixel:.3f} m/pixel | "
              f"Coverage: {640*metres_per_pixel:.0f}m x {640*metres_per_pixel:.0f}m")
    plt.axis('off')
    plt.show()

else:
    # If something went wrong, print the status code and message
    # Common issues: wrong API key, API not enabled, coordinates invalid
    print(f"Request failed with status code: {response.status_code}")
    print(f"Error message: {response.text}")