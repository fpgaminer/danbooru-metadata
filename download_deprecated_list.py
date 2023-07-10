#!/usr/bin/env python3
import requests

url = "https://danbooru.donmai.us/tags.json?commit=Search&search[hide_empty]=yes&search[is_deprecated]=yes&search[order]=date&limit=1000"

response = requests.get(url)
response.raise_for_status()

response = response.json()
if len(response) >= 1000:
	print("WARNING: There are more than 1000 deprecated tags. This script only downloads the first 1000.")

deprecated_tags = [tag["name"] for tag in response]
with open("tag_deprecations.txt", "w") as f:
	f.write("\n".join(deprecated_tags))
