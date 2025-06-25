import json

# Path to your exported GUS work items JSON file
INPUT_FILE = 'all_gus_work_items.json'  # Change this if your file is named differently

target_teams = {"AP Galaxy", "GLXY-Activities"}
target_types = {"Investigation", "Bug"}  # Adjust if 'Investigation' is a different type

try:
    with open(INPUT_FILE) as f:
        work_items = json.load(f)
except FileNotFoundError:
    print(f"File '{INPUT_FILE}' not found. Please export your GUS work items as JSON and place them in the project directory.")
    exit(1)

investigations = [
    wi for wi in work_items
    if wi.get("ScrumTeam") in target_teams and wi.get("Type") in target_types
]

if not investigations:
    print("No investigations found for the specified teams.")
else:
    print(f"Found {len(investigations)} investigations for teams {', '.join(target_teams)}:")
    for inv in investigations:
        print(f"- {inv.get('Name')} | {inv.get('Subject')} | {inv.get('Status')} | {inv.get('Link')}") 