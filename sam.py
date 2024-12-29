

import phonenumbers
from phonenumbers import geocoder
from opencage.geocoder import OpenCageGeocode
import folium

Key = "6d6f969fd9024ac8afde957f0c86a5ba"  # Replace this with your OpenCage API key

# Initialize OpenCage geocoder
geocoder = OpenCageGeocode(Key)

# Function to get location for a phone number
def get_location(phone_number):
    check_number = phonenumbers.parse(phone_number)
    number_location = geocoder.description_for_number(check_number, "en")
    print(number_location)

    service_provider = phonenumbers.parse(phone_number)
    print(carrier.name_for_number(service_provider, "en"))

    results = geocoder.geocode(number_location)
    if results:
        lat = results[0]['geometry']['lat']
        lng = results[0]['geometry']['lng']
        print(lat, lng)

        map_location = folium.Map(location=[lat, lng], zoom_start=9)
        folium.Marker([lat, lng], popup=number_location).add_to(map_location)
        map_location.save(f"{phone_number}_location.html")  # Save each location with a unique filename

# Example phone numbers to track
phone_numbers = ["+1234567890", "+1987654321", "+1122334455"]  # Replace with actual phone numbers

# Get location for each phone number
for number in phone_numbers:
    get_location(number)