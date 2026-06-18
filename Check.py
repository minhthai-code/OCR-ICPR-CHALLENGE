import os
import re
import webbrowser

def generate_error_gallery():
    # --- CONFIGURE THESE PATHS ---
    SCENARIO_ROOT = r"D:\AIOT PROJET\OCR-MultiFrame-ICPR\data\train\Scenario-B"
    LOG_FILE_PATH = r"D:\AIOT PROJET\error.csv"   # change if your file is named differently
    HTML_OUTPUT = "error_gallery.html"

    # --- Validate log file ---
    if not os.path.exists(LOG_FILE_PATH):
        print(f"❌ Error: Cannot find the log file at '{LOG_FILE_PATH}'.")
        print("Please place the correct error file in that location.")
        return

    print(f"📥 Loading '{LOG_FILE_PATH}'...")
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        log_content = f.read()

    # Regex: captures track ID, GT, and Prediction (plate)
    pattern = r"Track\s+(track_\d+)\s+\|\s+GT:\s+(\w+)\s+\|\s+Pred:\s+(\w+)"
    matches = re.findall(pattern, log_content)

    if not matches:
        print("❌ Parsing failed: No targets found matching the log pattern format.")
        print("Make sure the file contains lines like:")
        print('   1. Track track_10011 | GT: ODE3320 | Pred: DQE3320 (conf=0.5341) | ...')
        return

    print(f"🔍 Found {len(matches)} misclassifications. Building gallery...")

    html_cards = []
    for track_id, gt, pred in matches:
        # Try both layout folders
        img_path = ""
        for layout in ["Brazilian", "Mercosur"]:
            test_path = os.path.join(SCENARIO_ROOT, layout, track_id, "hr-001.jpg")
            if os.path.exists(test_path):
                img_path = test_path
                break

        if not img_path:
            print(f"⚠️  Image not found for {track_id}, skipping.")
            continue

        # Convert to URL-friendly path
        clean_url = img_path.replace("\\", "/")

        card = f"""
        <div style="background-color: #1e1e1e; border: 1px solid #333; border-radius: 6px; padding: 12px; display: flex; gap: 15px; align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="flex: 1; min-width: 180px;">
                <h4 style="margin: 0 0 8px 0; color: #ff6b6b; font-family: sans-serif;">📂 {track_id}</h4>
                <p style="margin: 4px 0; font-family: sans-serif; font-size: 14px; color: #aaa;"><b>GT:</b> <span style="color: #4da6ff; font-family: monospace; font-size: 16px; font-weight: bold;">{gt}</span></p>
                <p style="margin: 4px 0; font-family: sans-serif; font-size: 14px; color: #aaa;"><b>Pred:</b> <span style="color: #ff4d4d; font-family: monospace; font-size: 16px; font-weight: bold;">{pred}</span></p>
            </div>
            <div style="flex: 2; text-align: right;">
                <img src="file:///{clean_url}" style="max-height: 90px; max-width: 100%; border-radius: 4px; border: 1px solid #444;" alt="Plate crop">
            </div>
        </div>
        """
        html_cards.append(card)

    if not html_cards:
        print("❌ No valid image files found for any matched tracks.")
        return

    # Build full HTML page
    html_document = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>OCR Error Review Gallery</title>
    </head>
    <body style="background-color: #121212; margin: 0; padding: 20px; font-family: sans-serif;">
        <h2 style="color: #ffffff; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px;">❌ OCR Validation Failures ({len(html_cards)} Cases)</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 15px;">
            {"".join(html_cards)}
        </div>
    </body>
    </html>
    """

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_document)

    full_html_path = os.path.abspath(HTML_OUTPUT)
    print(f"✅ Gallery generated: {full_html_path}")
    webbrowser.open(full_html_path)   # Opens in your default browser

if __name__ == "__main__":
    generate_error_gallery()