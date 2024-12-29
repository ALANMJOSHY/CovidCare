import geopy
from geopy.geocoders import Nominatim

# Initialize the geolocator
geolocator = Nominatim(user_agent="my_geocoder")

# Define latitude and longitude
latitude = 9.412
longitude =76.41

# Get the address from latitude and longitude
location = geolocator.reverse((latitude, longitude), language="en")

# Print the address
print(f"Address for latitude {latitude} and longitude {longitude}: {location.address}")
