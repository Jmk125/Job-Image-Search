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
          "You are a construction-site photo describer. Describe what construction activity, materials, equipment, and stage of work are visible.",
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
              "Describe this construction scene in one concise sentence. Focus on the main activity (e.g., foundation work, masonry, steel erection, roofing), visible equipment (e.g., excavator, scaffolding, lifts), and materials.",
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
      const embedding = await embedText(description);

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

// Simple UI
app.get("/", (req, res) => {
  res.send(`
    <!doctype html>
    <html>
      <head>
        <title>Job Photo Search</title>
        <style>
          body { font-family: system-ui, sans-serif; margin: 20px; }
          h1 { margin-bottom: 0.5rem; }
          h2 { margin-top: 2rem; }
          label { display: block; margin-top: 8px; }
          input[type="text"], input[type="date"] { padding: 6px; width: 300px; }
          #results { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-top: 20px; }
          .photo-card { border: 1px solid #ddd; padding: 8px; border-radius: 6px; background: #fafafa; }
          .photo-card img { max-width: 100%; display: block; border-radius: 4px; }
          .meta { font-size: 12px; color: #555; margin-top: 4px; }
          button { padding: 6px 12px; margin-top: 8px; cursor: pointer; }
          .section { border: 1px solid #ddd; padding: 12px; border-radius: 8px; margin-bottom: 20px; }
        </style>
      </head>
      <body>
        <h1>Job Photo Search</h1>

        <div class="section">
          <h2>Upload photos to a project</h2>
          <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
            <label>
              Project name:
              <input type="text" name="project" placeholder="e.g. NorthRidgevilleHS" required />
            </label>
            <label>
              Shot date (optional, defaults to today):
              <input type="date" name="shot_date" />
            </label>
            <label>
              Photos:
              <input type="file" name="photos" multiple accept="image/*" required />
            </label>
            <button type="submit">Upload & Index</button>
          </form>
        </div>

        <div class="section">
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

        <script>
          const searchForm = document.getElementById('searchForm');
          const resultsDiv = document.getElementById('results');

          searchForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const q = document.getElementById('q').value;
            const project = document.getElementById('projectFilter').value;
            const shotDate = document.getElementById('shotDateFilter').value;

            resultsDiv.innerHTML = 'Searching...';

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
                <img src="\${r.image_url}" alt="">
                <div class="meta">
                  <strong>\${r.project || 'Unknown project'}</strong><br/>
                  \${r.shot_date || ''}<br/>
                  <em>\${r.description}</em><br/>
                  Score: \${r.score.toFixed(3)}
                </div>
              \`;
              resultsDiv.appendChild(card);
            });

            if (!data.results.length) {
              resultsDiv.innerHTML = '<p>No strong matches found for that search.</p>';
            }
          });
        </script>
      </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log("Server listening on http://localhost:" + PORT);
});
