import os
import glob
import json
from bs4 import BeautifulSoup

# Find all the text files you created
txt_files = glob.glob("../data/raw/*.txt")
print(txt_files)

for file_path in txt_files:
    print(f"Processing {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if not lines:
        continue
        
    # 1. The first line is your URL
    url = lines[0].strip()
    
    # 2. The rest of the lines are raw HTML
    html_content = "".join(lines[1:])
    
    # 3. Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 4. Clean up noise! 
    # Remove <style> tags (like the tab CSS) and <svg> icons (clipboard icons)
    for junk in soup(["style", "script", "svg"]):
        junk.extract()
        
    # 5. Extract the Title dynamically
    # Look for the first <h1> tag. If it doesn't exist, use the filename.
    h1_tag = soup.find('h1')
    if h1_tag:
        title = h1_tag.get_text(strip=True)
    else:
        title = f"Watsonx Document - {os.path.basename(file_path).replace('.txt', '')}"
        
    # 6. Extract the clean text
    # Using separator=' ' ensures that elements like <li> don't mash their words together
    text_content = soup.get_text(separator=' ', strip=True)
    
    # 7. Structure the JSON payload
    doc_data = {
        "title": title,
        "source": url,
        "category": "Gen AI Solutions",
        "text": text_content
    }
    
    # 8. Save as a JSON file
    json_filename = file_path.replace(".txt", ".json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(doc_data, f, indent=4)
        
    print(f"Saved formatted JSON to {json_filename}")

print("\nAll data structured! Layer 1 is officially complete.")