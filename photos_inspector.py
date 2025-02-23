import os
import shutil
import threading
import time
import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image
from flask import Flask, request, send_from_directory, render_template_string, jsonify
from urllib.parse import unquote

# ---------------------------
# Global Variables and Config
# ---------------------------
MODEL_OPTIONS = {
    "SigLIP2 Base": "google/siglip2-base-patch16-512",
    "SigLIP2 Large": "google/siglip2-large-patch16-512"
}
processing = False
total_images = 0
processed_images = 0
embeddings = {}  # filepath -> embedding vector
BASE_FOLDER = None  # will be set from user input
start_time = None  # time when processing started
model = None
processor = None
EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')  # default extensions

# ---------------------------
# Helper: Case-insensitive file search
# ---------------------------
def find_file_case_insensitive(directory, filename):
    for f in os.listdir(directory):
        if f.lower() == filename.lower():
            return f
    return None

# ---------------------------
# Helper Function: Process Images in Background
# ---------------------------
def process_images():
    global processing, total_images, processed_images, embeddings, BASE_FOLDER, start_time, EXTENSIONS
    image_files = []
    for root, dirs, files in os.walk(BASE_FOLDER):
        for file in files:
            if file.lower().endswith(EXTENSIONS):
                image_files.append(os.path.join(root, file))
    total_images = len(image_files)
    processed_images = 0
    embeddings = {}
    print(f"Found {total_images} images to process.")
    start_time = time.time()  # record start time
    for file in image_files:
        try:
            image = load_image(file)
            inputs = processor(images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
            emb = emb.cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)
            embeddings[file] = emb
            processed_images += 1
            print(f"Processed: {file} ({processed_images}/{total_images})")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    processing = False

# ---------------------------
# Clustering Function
# ---------------------------
def compute_clusters(threshold: float):
    files = list(embeddings.keys())
    n = len(files)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(embeddings[files[i]], embeddings[files[j]])
            if sim >= threshold:
                union(i, j)
    clusters_map = {}
    for i, file in enumerate(files):
        root = find(i)
        clusters_map.setdefault(root, []).append(file)
    clusters = [cluster for cluster in clusters_map.values() if len(cluster) > 1]
    return sorted(clusters, key=lambda cluster: len(cluster))

# ---------------------------
# Unclustered Images Function
# ---------------------------
def compute_unclustered(threshold: float):
    all_files = set(embeddings.keys())
    clusters = compute_clusters(threshold)
    clustered_files = set()
    for cluster in clusters:
        clustered_files.update(cluster)
    unclustered = list(all_files - clustered_files)
    # Removed default sort to allow dynamic sorting in route
    return unclustered

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__)

@app.route("/image/<path:filename>")
def serve_image(filename):
    safe_filename = unquote(filename)
    return send_from_directory(BASE_FOLDER, safe_filename)

@app.route("/start", methods=["POST"])
def start():
    global processing, BASE_FOLDER, model, processor, EXTENSIONS
    folder = request.form.get("folder")
    selected_model = request.form.get("model", MODEL_OPTIONS["SigLIP2 Base"])
    if not folder or not os.path.isdir(folder):
        return "Invalid folder path."
    BASE_FOLDER = os.path.abspath(folder)
    # Load selected model and processor
    model = AutoModel.from_pretrained(selected_model, device_map="auto", local_files_only=True).eval()
    processor = AutoImageProcessor.from_pretrained(selected_model, local_files_only=True)
    # Process additional extensions
    default_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    extensions_str = request.form.get("extensions")
    if extensions_str:
        extra = [x.strip() if x.strip().startswith('.') else '.'+x.strip() for x in extensions_str.split(',') if x.strip()]
        EXTENSIONS = default_exts + tuple(extra)
    else:
        EXTENSIONS = default_exts
    print(f"{EXTENSIONS = }")
    processing = True
    threading.Thread(target=process_images, daemon=True).start()
    return "Processing started."

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

@app.route("/progress")
def progress():
    global total_images, processed_images, processing, start_time
    pct = int((processed_images / total_images) * 100) if total_images else 0
    time_left = ""
    if processed_images > 0 and total_images:
        elapsed = time.time() - start_time
        avg = elapsed / processed_images
        remaining = (total_images - processed_images) * avg
        time_left = format_time(remaining)
    return jsonify({"progress": pct, "processing": processing, "time_left": time_left})

@app.route("/export", methods=["POST"])
def export():
    global BASE_FOLDER
    data = request.get_json()
    selected = data.get("selected", [])
    if not selected:
        return jsonify({"status": "No images selected."}), 400
    export_folder = BASE_FOLDER + "_copy"
    failed = []
    for rel_path in selected:
        rel_path_decoded = unquote(rel_path)
        src = os.path.join(BASE_FOLDER, rel_path_decoded)
        dst = os.path.join(export_folder, rel_path_decoded)
        if not os.path.exists(src):
            dir_name = os.path.dirname(src)
            base_name = os.path.basename(src)
            alt = find_file_case_insensitive(dir_name, base_name)
            if alt:
                src = os.path.join(dir_name, alt)
            else:
                print(f"File not found: {src}")
                failed.append(src)
                continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Error copying {src}: {e}")
            failed.append(src)
    return jsonify({"status": "Export completed.", "export_folder": export_folder, "failed": failed})

@app.route("/clusters")
def clusters_route():
    try:
        threshold = float(request.args.get("threshold", 0.8))
    except ValueError:
        threshold = 0.8
    clusters = compute_clusters(threshold)
    cluster_count = len(clusters)
    html_content = f"<h3>Number of groups: {cluster_count}</h3>"
    if clusters:
        for i, cluster in enumerate(clusters, start=1):
            count = len(cluster)
            html_content += f"<div style='margin-bottom:20px; padding:10px; border:1px solid #ccc;'>"
            html_content += f"<p>Group #{i} has {count} elements</p>"
            for idx, img in enumerate(cluster):
                rel_path = os.path.relpath(img, BASE_FOLDER)
                url = f"/image/{rel_path}"
                cls = "selected" if idx == 0 else ""
                html_content += f"<img src='{url}' data-rel='{rel_path}' title='{rel_path}' class='{cls}' style='height:200px; margin-right:10px; cursor:pointer;' onclick='handleClick(event, this)' ondblclick='handleDblClick(event, this)'/>"
            html_content += "</div>"
    else:
        html_content += "<p>No groups found for this threshold.</p>"
    return html_content

@app.route("/unclustered")
def unclustered_route():
    try:
        threshold = float(request.args.get("threshold", 0.8))
    except ValueError:
        threshold = 0.8
    sort_by = request.args.get("sort_by", "name")
    unclustered = compute_unclustered(threshold)
    if sort_by == "date_of_creation":
        unclustered.sort(key=lambda x: os.path.getctime(x))
    else:
        unclustered.sort(key=lambda x: os.path.basename(x).lower())
    html_content = f"<h3>Images without a group: {len(unclustered)}</h3>"
    if unclustered:
        for img in unclustered:
            rel_path = os.path.relpath(img, BASE_FOLDER)
            url = f"/image/{rel_path}"
            html_content += f"<img src='{url}' data-rel='{rel_path}' title='{rel_path}' class='selected' style='height:200px; margin-right:10px; cursor:pointer;' onclick='handleClick(event, this)' ondblclick='handleDblClick(event, this)'/>"
    else:
        html_content += "<p>No ungrouped images found.</p>"
    return html_content

@app.route("/")
def index():
    html_page = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Local images groups by similarity</title>
        <link href="https://fonts.googleapis.com/css?family=Roboto:400,500,700" rel="stylesheet">
        <style>
          body {
            font-family: 'Roboto', sans-serif;
            background-color: #fafafa;
            color: #333;
            margin: 0;
            padding: 20px;
          }
          h1 {
            color: #2c3e50;
          }
          label, p {
            font-size: 1.1em;
          }
          input[type="text"], select, input[type="range"] {
            padding: 8px;
            font-size: 1em;
            margin: 5px 0 10px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
          }
          button {
            padding: 8px 16px;
            font-size: 1em;
            background-color: #3498db;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
          }
          button:hover {
            background-color: #2980b9;
          }
          #progressBar {
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
          }
          #progressBar div {
            width: 0%;
            height: 20px;
            background-color: #3498db;
            text-align: center;
            color: #fff;
          }
          .group-container, .unclustered-container {
            margin-top: 20px;
          }
          .group-container > div {
            background: #fff;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
          }
          img.selected {
            border: 3px solid #e74c3c;
          }
          /* Modal overlay styles */
          .modal {
            display: none; 
            position: fixed; 
            z-index: 1000; 
            left: 0;
            top: 0;
            width: 100%; 
            height: 100%; 
            overflow: auto; 
            background-color: rgba(0,0,0,0.8);
          }
          .modal-content {
            margin: 5% auto;
            display: block;
            max-width: 90%;
            border-radius: 4px;
          }
          .modal-close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
          }
          #timeLeft, #selectedCount {
            font-weight: bold;
          }
          /* Hide ungrouped images by default */
          #unclusteredSection {
            display: none;
          }
          .toggle-btn, .toggle-select-btn {
            background-color: #2ecc71;
            margin-top: 10px;
          }
        </style>
      </head>
      <body>
        <h1>Local images groups by similarity</h1>
        <!-- Folder Path and Model Selection Form -->
        <form id="folderForm" method="post" action="/start">
          <label for="folder">Enter folder path:</label>
          <input type="text" id="folder" name="folder" value="/Users/daniil/Downloads/2005-2009 (4)" required>
          <button type="button" onclick="openInFinder()">Open in Finder</button>
          <input type="file" id="folderPicker" webkitdirectory style="display:none;" onchange="handleFolderSelect(event)">
          <br>
          <label for="extensions">Additional extensions (comma separated, e.g., .tiff, .webp):</label>
          <input type="text" id="extensions" name="extensions" placeholder=".tiff, .webp">
          <br>
          <label for="model">Select model:</label>
          <select id="model" name="model">
            <option value="google/siglip2-base-patch16-512">SigLIP2 Base</option>
            <option value="google/siglip2-large-patch16-512">SigLIP2 Large</option>
          </select>
          <br>
          <button type="submit">Start Processing</button>
        </form>
        <!-- Progress Bar and Time Remaining -->
        <div id="progressContainer" style="display:none;">
          <h3>Processing Images... (<span id="timeLeft">--:--</span> remaining)</h3>
          <div id="progressBar"><div id="progressBarFill">0%</div></div>
        </div>
        <!-- Clusters and Unclustered UI -->
        <div id="clustersSection" style="display:none;">
          <label for="threshold">Similarity Threshold: <span id="thresh_val">0.80</span></label>
          <input type="range" id="threshold" min="0.5" max="1.0" step="0.01" value="0.80" onchange="debouncedUpdateClusters(this.value)">
          <button onclick="copySelected()">Copy Selected URLs</button>
          <button onclick="exportSelected()">Export Selected Files</button>
          <span id="selectedCount">Selected: 0</span>
          <div id="clusters" class="group-container" style="margin-top:20px;"></div>
          <hr>
          <h3>Images without a group</h3>
          <label for="sortBy">Sort by:</label>
          <select id="sortBy" onchange="debouncedUpdateClusters(document.getElementById('threshold').value)">
            <option value="name">Name</option>
            <option value="date_of_creation">Date of Creation</option>
          </select>
          <button class="toggle-btn" onclick="toggleUnclustered()">Show/Hide</button>
          <button class="toggle-select-btn" onclick="toggleUnclusteredSelect()">Select All / Unselect All</button>
          <div id="unclusteredSection" class="unclustered-container" style="margin-top:20px;"></div>
        </div>
        <!-- Modal for larger preview -->
        <div id="imgModal" class="modal" onclick="closeModal()">
          <span class="modal-close" onclick="closeModal()">&times;</span>
          <img class="modal-content" id="modalImage">
        </div>
        <script>
          var clickTimeout = null;
          function handleClick(event, img) {
            if (clickTimeout !== null) return;
            clickTimeout = setTimeout(function() {
              toggleSelect(img);
              updateSelectedCount();
              clickTimeout = null;
            }, 250);
          }
          function handleDblClick(event, img) {
            clearTimeout(clickTimeout);
            clickTimeout = null;
            showPreview(img);
          }
          function toggleSelect(img) {
            img.classList.toggle("selected");
          }
          function showPreview(img) {
            var modal = document.getElementById("imgModal");
            var modalImg = document.getElementById("modalImage");
            modal.style.display = "block";
            modalImg.src = img.src;
          }
          function closeModal() {
            document.getElementById("imgModal").style.display = "none";
          }
          function updateSelectedCount() {
            let count = document.querySelectorAll("img.selected").length;
            document.getElementById("selectedCount").innerText = "Selected: " + count;
          }
          function copySelected() {
            let selectedImgs = document.querySelectorAll("img.selected");
            if(selectedImgs.length === 0) {
              alert("No images selected.");
              return;
            }
            let urls = [];
            selectedImgs.forEach(img => {
              urls.push(img.getAttribute("data-rel"));
            });
            navigator.clipboard.writeText(urls.join("\\n")).then(() => {
              alert("Copied " + urls.length + " image URL(s) to clipboard.");
            });
          }
          function exportSelected() {
            let selectedImgs = document.querySelectorAll("img.selected");
            if(selectedImgs.length === 0) {
              alert("No images selected.");
              return;
            }
            let relPaths = [];
            selectedImgs.forEach(img => {
              relPaths.push(img.getAttribute("data-rel"));
            });
            fetch("/export", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({selected: relPaths})
            }).then(response => response.json())
              .then(data => { 
                let message = "Export completed to: " + data.export_folder;
                if(data.failed && data.failed.length > 0) {
                  message += "\\nFailed to copy the following files:\\n" + data.failed.join("\\n");
                }
                alert(message);
              });
          }
          function toggleUnclusteredSelect() {
            let imgs = document.querySelectorAll("#unclusteredSection img");
            let allSelected = true;
            imgs.forEach(img => {
              if (!img.classList.contains("selected")) {
                allSelected = false;
              }
            });
            imgs.forEach(img => {
              if (allSelected) {
                img.classList.remove("selected");
              } else {
                img.classList.add("selected");
              }
            });
            updateSelectedCount();
          }
          function toggleUnclustered() {
            let section = document.getElementById("unclusteredSection");
            if (section.style.display === "none") {
              section.style.display = "block";
            } else {
              section.style.display = "none";
            }
          }
          function debouncedUpdateClusters(val) {
            document.getElementById("thresh_val").innerText = parseFloat(val).toFixed(2);
            clearTimeout(clickTimeout);
            clickTimeout = null;
            setTimeout(() => {
              let sortBy = document.getElementById("sortBy").value;
              fetch("/clusters?threshold=" + val)
                .then(response => response.text())
                .then(data => {
                  document.getElementById("clusters").innerHTML = data;
                  updateSelectedCount();
                });
              fetch("/unclustered?threshold=" + val + "&sort_by=" + sortBy)
                .then(response => response.text())
                .then(data => {
                  document.getElementById("unclusteredSection").innerHTML = data;
                  updateSelectedCount();
                });
            }, 300);
          }
          function openInFinder() {
            document.getElementById("folderPicker").click();
          }
          function handleFolderSelect(event) {
            var files = event.target.files;
            if(files.length > 0) {
              // Extract the folder path from the first file's webkitRelativePath
              var path = files[0].webkitRelativePath;
              // Remove the file name from the path to get the folder
              var folder = path.split("/")[0];
              document.getElementById("folder").value = folder;
            }
          }
          document.getElementById("folderForm").addEventListener("submit", function(e) {
            e.preventDefault();
            var formData = new FormData(this);
            fetch("/start", { method: "POST", body: formData })
              .then(response => response.text())
              .then(data => {
                alert(data);
                document.getElementById("folderForm").style.display = "none";
                document.getElementById("progressContainer").style.display = "block";
                pollProgress();
              });
          });
          function pollProgress() {
            var interval = setInterval(() => {
              fetch("/progress")
                .then(response => response.json())
                .then(data => {
                  document.getElementById("progressBarFill").style.width = data.progress + "%";
                  document.getElementById("progressBarFill").innerText = data.progress + "%";
                  document.getElementById("timeLeft").innerText = data.time_left ? data.time_left : "--:--";
                  if (!data.processing) {
                    clearInterval(interval);
                    document.getElementById("progressContainer").style.display = "none";
                    document.getElementById("clustersSection").style.display = "block";
                    debouncedUpdateClusters(document.getElementById("threshold").value);
                  }
                });
            }, 500);
          }
        </script>
      </body>
    </html>
    """
    return render_template_string(html_page)

if __name__ == "__main__":
    app.run(debug=True)