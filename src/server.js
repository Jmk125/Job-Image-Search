const express = require("express");
const path = require("path");
const fs = require("fs");
const sqlite3 = require("sqlite3").verbose();
const multer = require("multer");
const dotenv = require("dotenv");
const fetch = require("node-fetch"); // direct HTTP calls to OpenAI

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3040;

const DB_PATH = path.join(process.cwd(), "data", "photo_index.db");
const IMAGES_ROOT = path.join(process.cwd(), "data", "images_root");

// Ensure images_root exists
if (!fs.existsSync(IMAGES_ROOT)) {
  fs.mkdirSync(IMAGES_ROOT, { recursive: true });
}

// ---------- DB SETUP (sqlite3, async) ----------

if (!fs.existsSync(DB_PATH)) {
  fs.writeFileSync(DB_PATH, "");
}

const db = new sqlite3.Database(DB_PATH);

db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS photos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_path TEXT UNIQUE,
      project TEXT,
      shot_date TEXT,
      description TEXT,
      embedding BLOB
    );
  `);
});

// Project summary helper
function getProjectsSummary() {
  return new Promise((resolve, reject) => {
    db.all(
      `
      SELECT project, COUNT(*) as count, MIN(shot_date) as earliest, MAX(shot_date) as latest
      FROM photos
      GROUP BY project
      ORDER BY project COLLATE NOCASE
    `,
      [],
      (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      }
    );
  });
}

// Insert helper
function insertPhoto(record) {
  return new Promise((resolve, reject) => {
    db.run(
      `
      INSERT OR IGNORE INTO photos
      (file_path, project, shot_date, description, embedding)
      VALUES (?, ?, ?, ?, ?)
    `,
      [
        record.file_path,
        record.project,
        record.shot_date,
        record.description,
        record.embedding,
      ],
      function (err) {
        if (err) reject(err);
        else resolve();
      }
    );
  });
}

// Fetch helper (with optional filters)
function getAllPhotos(projectFilter, dateFilter) {
  return new Promise((resolve, reject) => {
    let sql = "SELECT * FROM photos WHERE 1=1";
    const params = [];

    if (projectFilter) {
      sql += " AND project = ?";
      params.push(projectFilter);
    }
    if (dateFilter) {
      sql += " AND shot_date = ?";
      params.push(dateFilter);
    }

    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
}

function getPhotoById(id) {
  return new Promise((resolve, reject) => {
    db.get("SELECT * FROM photos WHERE id = ?", [id], (err, row) => {
      if (err) reject(err);
      else resolve(row);
    });
  });
}

// ---------- Helpers ----------

function todayString() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function fileToBase64(filePath) {
  const data = fs.readFileSync(filePath);
  const ext = path.extname(filePath).toLowerCase().replace(".", "") || "jpeg";
  return `data:image/${ext};base64,` + data.toString("base64");
}

// Call OpenAI chat completions with vision via HTTP
async function describeImageWithAI(imagePath) {
  const base64 = fileToBase64(imagePath);

  const body = {
    model: "gpt-4.1-mini", // vision-capable model
    messages: [
      {
        role: "system",
        content:
          "You are a construction progress inspector who writes terse, technical captions. Identify trade (e.g., concrete, structural steel, roofing, MEP rough-in), primary activity, major equipment, materials, location context, and stage of completion.",
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
              "Provide a single sentence: start with trade + activity, then materials/equipment, and any QA/QC notes (formwork status, rebar spacing, welds, waterproofing, PPE).",
          },
          {
            type: "image_url",
            image_url: { url: base64 },
          },
        ],
      },
    ],
    max_tokens: 80,
  };

  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error("OpenAI vision error:", resp.status, errText);
    throw new Error("OpenAI vision request failed");
  }

  const json = await resp.json();
  const text = json.choices?.[0]?.message?.content?.trim();
  return text || "Construction site photo.";
}

// Call OpenAI embeddings via HTTP
async function getEmbedding(text) {
  const body = {
    model: "text-embedding-3-small",
    input: text,
  };

  const resp = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error("OpenAI embeddings error:", resp.status, errText);
    throw new Error("OpenAI embeddings request failed");
  }

  const json = await resp.json();
  return json.data?.[0]?.embedding;
}

async function embedText(text) {
  const vector = await getEmbedding(text);
  const floatArray = new Float32Array(vector);
  return Buffer.from(floatArray.buffer);
}

async function embedQuery(text) {
  return getEmbedding(text);
}

function blobToVector(blob) {
  if (!blob) return null;
  const buffer = Buffer.from(blob);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function deletePhotoRecord(id) {
  return new Promise((resolve, reject) => {
    db.run("DELETE FROM photos WHERE id = ?", [id], function (err) {
      if (err) reject(err);
      else resolve();
    });
  });
}

// ---------- Multer upload config ----------

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const projectRaw = req.body.project || "UnknownProject";
    const project = projectRaw.replace(/[^a-zA-Z0-9_-]/g, "_");
    const dateFolder = req.body.shot_date || todayString();

    const dest = path.join(IMAGES_ROOT, project, dateFolder);
    fs.mkdirSync(dest, { recursive: true });
    cb(null, dest);
  },
  filename: function (req, file, cb) {
    const ts = Date.now();
    const ext = path.extname(file.originalname) || ".jpg";
    const cleanName = file.originalname.replace(/\s+/g, "_");
    cb(null, `${ts}-${cleanName}${ext}`);
  },
});

const upload = multer({ storage });

// ---------- Middleware / static ----------

app.use(express.urlencoded({ extended: true }));
app.use("/images", express.static(IMAGES_ROOT));

// ---------- Routes ----------

// Upload & index route
app.post("/upload", upload.array("photos", 300), async (req, res) => {
  const projectRaw = req.body.project || "UnknownProject";
  const project = projectRaw.replace(/[^a-zA-Z0-9_-]/g, "_");
  const shot_date = req.body.shot_date || todayString();
  const files = req.files || [];

  if (!files.length) {
    return res.status(400).send("No files uploaded.");
  }

  for (const file of files) {
    try {
      const absPath = file.path;
      const relativePath = path.relative(IMAGES_ROOT, absPath);

      const description = await describeImageWithAI(absPath);
      const embeddingText = `${description} | Project:${project} | Shot:${shot_date}`;
      const embedding = await embedText(embeddingText);

      await insertPhoto({
        file_path: relativePath,
        project,
        shot_date,
        description,
        embedding,
      });

      console.log(`Indexed uploaded photo: ${relativePath}`);
    } catch (err) {
      console.error(
        "Error indexing uploaded file:",
        file.filename,
        err.message
      );
    }
  }

  res.send(`
    <html>
      <body>
        <p>Uploaded and indexed ${files.length} photo(s) for project <strong>${project}</strong>.</p>
        <p><a href="/">Back to search</a></p>
      </body>
    </html>
  `);
});

// Search route
app.get("/search", async (req, res) => {
  const query = req.query.q;
  const limit = parseInt(req.query.limit || "20", 10);
  const projectFilter = req.query.project || "";
  const dateFilter = req.query.shot_date || "";

  if (!query) {
    return res.status(400).json({ error: "Missing 'q' query parameter" });
  }

  try {
    const queryEmbedding = await embedQuery(query);
    const qVec = Float32Array.from(queryEmbedding);

    const rows = await getAllPhotos(projectFilter, dateFilter);

    const scored = rows
      .map((row) => {
        const embVec = blobToVector(row.embedding);
        if (!embVec) return null;
        const score = cosineSimilarity(qVec, embVec);
        return { ...row, score };
      })
      .filter(Boolean);

    // Sort best → worst
    scored.sort((a, b) => b.score - a.score);

    // 🔹 Only keep “good enough” matches
    const MIN_SCORE = 0.30; // tweak this to adjust strictness
    const filtered = scored.filter((row) => row.score >= MIN_SCORE);

    const top = filtered.slice(0, limit).map((row) => ({
      id: row.id,
      file_path: row.file_path,
      project: row.project,
      shot_date: row.shot_date,
      description: row.description,
      score: row.score,
      image_url: `/images/${row.file_path.replace(/\\/g, "/")}`,
    }));

    res.json({ query, results: top });
  } catch (err) {
    console.error("Search error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Project summary list
app.get("/projects", async (req, res) => {
  try {
    const projects = await getProjectsSummary();
    res.json({ projects });
  } catch (err) {
    console.error("Projects list error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Photos by project
app.get("/projects/:project/photos", async (req, res) => {
  const project = decodeURIComponent(req.params.project || "");
  if (!project) {
    return res.status(400).json({ error: "Missing project" });
  }

  try {
    const rows = await getAllPhotos(project, "");
    const ordered = rows
      .map((row) => ({
        ...row,
        image_url: `/images/${row.file_path.replace(/\\/g, "/")}`,
      }))
      .sort((a, b) => {
        if (a.shot_date === b.shot_date) return b.id - a.id;
        return (b.shot_date || "").localeCompare(a.shot_date || "");
      });

    res.json({ project, photos: ordered });
  } catch (err) {
    console.error("Project photos error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Delete photo
app.delete("/photos/:id", async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (!id) return res.status(400).json({ error: "Invalid id" });

  try {
    const row = await getPhotoById(id);
    if (!row) return res.status(404).json({ error: "Not found" });

    const fullPath = path.join(IMAGES_ROOT, row.file_path);
    if (fs.existsSync(fullPath)) {
      try {
        fs.unlinkSync(fullPath);
      } catch (err) {
        console.warn("Could not delete file", fullPath, err.message);
      }
    }

    await deletePhotoRecord(id);
    res.json({ ok: true });
  } catch (err) {
    console.error("Delete photo error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Simple UI
app.get("/", (req, res) => {
  res.send(`
    <!doctype html>
    <html>
      <head>
        <title>Job Photo Search</title>
        <style>
          :root {
            color-scheme: light;
            --accent: #4f46e5;
          }
          body {
            font-family: system-ui, sans-serif;
            margin: 0;
            padding: 24px;
            background: radial-gradient(circle at 10% 20%, #f5f9ff, #f8f8f8 45%);
            color: #1f2937;
          }
          h1 { margin-bottom: 0.25rem; letter-spacing: -0.02em; }
          h2 { margin-top: 1rem; }
          p.lead { margin-top: 0; color: #4b5563; }
          label { display: block; margin-top: 10px; font-weight: 600; }
          input[type="text"], input[type="date"], input[type="file"], select {
            padding: 10px;
            width: 100%;
            max-width: 420px;
            border-radius: 10px;
            border: 1px solid #d1d5db;
            margin-top: 4px;
            font-size: 14px;
          }
          button {
            padding: 10px 16px;
            margin-top: 12px;
            cursor: pointer;
            background: linear-gradient(135deg, #4f46e5, #6366f1);
            border: none;
            color: white;
            border-radius: 10px;
            font-weight: 700;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
            transition: transform 0.1s ease, box-shadow 0.1s ease;
          }
          button:hover { transform: translateY(-1px); box-shadow: 0 12px 22px rgba(79, 70, 229, 0.28); }
          button:active { transform: translateY(0); box-shadow: 0 6px 12px rgba(79, 70, 229, 0.2); }
          .shell { max-width: 1100px; margin: 0 auto; }
          .section { border: 1px solid #e5e7eb; padding: 18px; border-radius: 14px; margin-bottom: 18px; background: white; box-shadow: 0 12px 40px rgba(15, 23, 42, 0.05); }
          #results, #projectPhotos { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; margin-top: 20px; }
          .photo-card { border: 1px solid #e5e7eb; padding: 10px; border-radius: 12px; background: #fdfefe; box-shadow: inset 0 0 0 1px #f3f4f6; transition: transform 0.12s ease, box-shadow 0.12s ease; }
          .photo-card:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,0.08); }
          .photo-card img { max-width: 100%; display: block; border-radius: 10px; margin-bottom: 8px; cursor: pointer; }
          .meta { font-size: 13px; color: #4b5563; margin-top: 4px; line-height: 1.4; }
          .meta strong { color: #111827; }
          .score { font-size: 12px; color: #6b7280; }
          .actions { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 8px; }
          .pill { display: inline-block; padding: 4px 10px; background: #eef2ff; color: #4338ca; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.01em; }
          .tab-bar { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
          .tab-btn { padding: 10px 14px; border-radius: 999px; border: 1px solid #e5e7eb; background: #f3f4f6; color: #111827; font-weight: 700; cursor: pointer; }
          .tab-btn.active { background: linear-gradient(135deg, #4f46e5, #6366f1); color: white; border-color: transparent; box-shadow: 0 10px 30px rgba(79,70,229,0.25); }
          .tab-panel { display: none; }
          .tab-panel.active { display: block; }
          .status { margin-top: 10px; color: #374151; font-size: 14px; }
          .progress-shell { width: 100%; height: 12px; border-radius: 999px; background: #e5e7eb; overflow: hidden; margin-top: 8px; }
          .progress-bar { height: 100%; width: 0%; background: linear-gradient(135deg, #10b981, #22d3ee); transition: width 0.2s ease; }
          .inline-group { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
          .project-chip { padding: 10px 12px; border: 1px solid #e5e7eb; border-radius: 12px; background: #f8fafc; display: flex; justify-content: space-between; align-items: center; cursor: pointer; transition: box-shadow 0.12s ease; }
          .project-chip:hover { box-shadow: 0 8px 22px rgba(0,0,0,0.08); }
          .project-chip strong { color: #111827; }
          .soft { color: #6b7280; font-size: 13px; }
          .danger { background: linear-gradient(135deg, #ef4444, #f97316); box-shadow: 0 8px 20px rgba(239, 68, 68, 0.25); }
        </style>
      </head>
      <body>
        <div class="shell">
          <h1>Job Photo Search</h1>
          <p class="lead">Upload field photos, let AI label them, then search or browse by project.</p>

          <div class="tab-bar">
            <button class="tab-btn active" data-tab-target="uploadTab">Upload</button>
            <button class="tab-btn" data-tab-target="searchTab">Search</button>
            <button class="tab-btn" data-tab-target="projectsTab">Projects</button>
          </div>

          <div id="uploadTab" class="tab-panel active section">
            <h2>Upload photos</h2>
            <p class="soft">Choose an existing project or create a new one. We'll index the images as they're uploaded.</p>
            <div class="inline-group">
              <label>
                <input type="radio" name="projectMode" value="existing" checked /> Use existing project
              </label>
              <label>
                <input type="radio" name="projectMode" value="new" /> Create new project
              </label>
            </div>
            <form id="uploadForm" enctype="multipart/form-data">
              <label>
                Existing project:
                <select id="existingProject" name="existing_project"></select>
              </label>
              <label>
                New project name:
                <input type="text" id="newProject" name="new_project" placeholder="e.g. NorthRidgevilleHS" />
              </label>
              <label>
                Shot date (optional, defaults to today):
                <input type="date" id="shotDate" name="shot_date" />
              </label>
              <label>
                Photos:
                <input type="file" id="photos" name="photos" multiple accept="image/*" required />
              </label>
              <button type="submit" id="uploadBtn">Upload & Index</button>
              <div class="status" id="uploadStatus"></div>
              <div class="progress-shell"><div id="uploadProgress" class="progress-bar"></div></div>
            </form>
          </div>

          <div id="searchTab" class="tab-panel section">
            <h2>Search photos</h2>
            <form id="searchForm">
              <label>
                Search query:
                <input type="text" id="q" name="q" placeholder="e.g. scaffolding, footing rebar, excavator digging" required />
              </label>
              <label>
                Filter by project (optional):
                <input type="text" id="projectFilter" name="project" placeholder="exact project name" />
              </label>
              <label>
                Filter by shot date (optional):
                <input type="date" id="shotDateFilter" name="shot_date" />
              </label>
              <button type="submit">Search</button>
            </form>
            <div id="results"></div>
          </div>

          <div id="projectsTab" class="tab-panel section">
            <h2>Browse by project</h2>
            <p class="soft">See indexed projects, open images full-size, or delete unwanted shots.</p>
            <div id="projectsList" class="inline-group"></div>
            <div id="projectPhotos"></div>
          </div>
        </div>

        <script>
          const searchForm = document.getElementById('searchForm');
          const resultsDiv = document.getElementById('results');
          const uploadForm = document.getElementById('uploadForm');
          const uploadStatus = document.getElementById('uploadStatus');
          const uploadProgress = document.getElementById('uploadProgress');
          const uploadBtn = document.getElementById('uploadBtn');
          const projectsList = document.getElementById('projectsList');
          const projectPhotos = document.getElementById('projectPhotos');
          const existingProjectSelect = document.getElementById('existingProject');

          resultsDiv.innerHTML = '<p class="meta">Tip: search by trade + activity ("steel decking being welded", "CMU wall grouted", "waterproofing at podium") and narrow with project or date.</p>';

          function setActiveTab(targetId) {
            document.querySelectorAll('.tab-btn').forEach(btn => {
              btn.classList.toggle('active', btn.dataset.tabTarget === targetId);
            });
            document.querySelectorAll('.tab-panel').forEach(panel => {
              panel.classList.toggle('active', panel.id === targetId);
            });
            if (targetId === 'projectsTab') {
              loadProjects();
            }
          }

          document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => setActiveTab(btn.dataset.tabTarget));
          });

          async function loadProjects() {
            const resp = await fetch('/projects');
            const data = await resp.json();
            projectsList.innerHTML = '';
            existingProjectSelect.innerHTML = '';

            const defaultOpt = document.createElement('option');
            defaultOpt.value = '';
            defaultOpt.textContent = data.projects?.length ? 'Select a project' : 'No projects yet';
            existingProjectSelect.appendChild(defaultOpt);

            (data.projects || []).forEach(p => {
              const opt = document.createElement('option');
              opt.value = p.project;
              opt.textContent = p.project;
              existingProjectSelect.appendChild(opt);

              const chip = document.createElement('div');
              chip.className = 'project-chip';
              chip.innerHTML = \`
                <div>
                  <strong>\${p.project}</strong><br/>
                  <span class="soft">\${p.count} photo(s)</span>
                </div>
                <div class="soft">\${p.earliest || ''} - \${p.latest || ''}</div>
              \`;
              chip.addEventListener('click', () => loadProjectPhotos(p.project));
              projectsList.appendChild(chip);
            });
          }

          async function loadProjectPhotos(project) {
            if (!project) return;
            projectPhotos.innerHTML = '<p class="meta">Loading photos for ' + project + '...</p>';
            const encoded = encodeURIComponent(project);
            const resp = await fetch('/projects/' + encoded + '/photos');
            const data = await resp.json();

            projectPhotos.innerHTML = '';
            (data.photos || []).forEach(r => {
              const card = document.createElement('div');
              card.className = 'photo-card';
              card.innerHTML = \`
                <img src="\${r.image_url}" alt="Photo from \${r.project || 'project'}">
                <div class="meta">
                  <span class="pill">\${r.project || 'Unknown project'}</span><br/>
                  \${r.shot_date || 'Date unknown'}<br/>
                  <em>\${r.description || ''}</em>
                </div>
                <div class="actions">
                  <button type="button" data-url="\${r.image_url}">Open full-size</button>
                  <button type="button" data-id="\${r.id}" class="danger">Delete</button>
                </div>
              \`;
              const preview = card.querySelector('img');
              const openBtn = card.querySelector('button[data-url]');
              const deleteBtn = card.querySelector('button[data-id]');
              const openFull = () => window.open(r.image_url, '_blank', 'noopener');
              preview.addEventListener('click', openFull);
              openBtn.addEventListener('click', openFull);
              deleteBtn.addEventListener('click', async () => {
                deleteBtn.disabled = true;
                deleteBtn.textContent = 'Deleting...';
                const delResp = await fetch('/photos/' + r.id, { method: 'DELETE' });
                if (delResp.ok) {
                  card.remove();
                  loadProjects();
                } else {
                  deleteBtn.disabled = false;
                  deleteBtn.textContent = 'Delete';
                  alert('Failed to delete photo');
                }
              });
              projectPhotos.appendChild(card);
            });

            if (!data.photos?.length) {
              projectPhotos.innerHTML = '<p class="meta">No photos found for that project.</p>';
            }
          }

          uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            uploadStatus.textContent = '';
            uploadProgress.style.width = '0%';

            const mode = document.querySelector('input[name="projectMode"]:checked').value;
            const project = mode === 'existing'
              ? existingProjectSelect.value
              : document.getElementById('newProject').value.trim();

            if (!project) {
              uploadStatus.textContent = 'Please select or enter a project name.';
              return;
            }

            const photosInput = document.getElementById('photos');
            if (!photosInput.files.length) {
              uploadStatus.textContent = 'Please choose at least one photo.';
              return;
            }

            const fd = new FormData();
            fd.append('project', project);
            const shotDate = document.getElementById('shotDate').value;
            if (shotDate) fd.append('shot_date', shotDate);
            Array.from(photosInput.files).forEach(f => fd.append('photos', f));

            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload');
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Uploading...';

            xhr.upload.addEventListener('progress', (event) => {
              if (event.lengthComputable) {
                const pct = Math.round((event.loaded / event.total) * 100);
                uploadProgress.style.width = pct + '%';
                uploadStatus.textContent = 'Uploading ' + pct + '%';
              }
            });

            xhr.onload = () => {
              uploadBtn.disabled = false;
              uploadBtn.textContent = 'Upload & Index';
              uploadProgress.style.width = '100%';
              if (xhr.status >= 200 && xhr.status < 300) {
                uploadStatus.textContent = 'Upload complete. Indexing finished for ' + project + '.';
                photosInput.value = '';
                setActiveTab('projectsTab');
                loadProjects();
              } else {
                uploadStatus.textContent = 'Upload failed: ' + xhr.responseText;
              }
            };

            xhr.onerror = () => {
              uploadBtn.disabled = false;
              uploadBtn.textContent = 'Upload & Index';
              uploadStatus.textContent = 'Upload failed. Please retry.';
            };

            xhr.send(fd);
          });

          searchForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const q = document.getElementById('q').value;
            const project = document.getElementById('projectFilter').value;
            const shotDate = document.getElementById('shotDateFilter').value;

            resultsDiv.innerHTML = '<p class="meta">Searching...</p>';

            const params = new URLSearchParams();
            params.set('q', q);
            if (project) params.set('project', project);
            if (shotDate) params.set('shot_date', shotDate);

            const resp = await fetch('/search?' + params.toString());
            const data = await resp.json();

            resultsDiv.innerHTML = '';
            data.results.forEach(r => {
              const card = document.createElement('div');
              card.className = 'photo-card';
              card.innerHTML = \`
                <img src="\${r.image_url}" alt="Photo from \${r.project || 'project'}">
                <div class="meta">
                  <span class="pill">\${r.project || 'Unknown project'}</span><br/>
                  \${r.shot_date || 'Date unknown'}<br/>
                  <em>\${r.description}</em>
                </div>
                <div class="actions">
                  <span class="score">Score \${r.score.toFixed(3)}</span>
                  <button type="button" data-url="\${r.image_url}">Open full-size</button>
                </div>
              \`;
              const preview = card.querySelector('img');
              const openBtn = card.querySelector('button');
              const openFull = () => window.open(r.image_url, '_blank', 'noopener');
              preview.addEventListener('click', openFull);
              openBtn.addEventListener('click', openFull);
              resultsDiv.appendChild(card);
            });

            if (!data.results.length) {
              resultsDiv.innerHTML = '<p class="meta">No strong matches found for that search.</p>';
            }
          });

          loadProjects();
        </script>
      </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log("Server listening on http://localhost:" + PORT);
});
